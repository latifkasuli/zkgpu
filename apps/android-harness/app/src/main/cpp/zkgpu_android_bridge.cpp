#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace {

constexpr const char* kTag = "ZkgpuBridge";
constexpr const char* kRustLibrary = "libzkgpu_ffi.so";
constexpr int kFfiApiVersion = 1;

using RunRequestJsonFn = char* (*)(const char*);
using GetVersionJsonFn = char* (*)();
using FreeStringFn = void (*)(char*);

struct ZkgpuFfi {
    void* handle = nullptr;
    RunRequestJsonFn run_request_json = nullptr;
    GetVersionJsonFn get_version_json = nullptr;
    FreeStringFn free_string = nullptr;
    std::string error;
};

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

std::string error_json(const std::string& message) {
    return "{\"ok\":false,\"report\":null,\"error\":\"" + json_escape(message) + "\"}";
}

std::string version_error_json(const std::string& message) {
    return "{\"crate_name\":\"zkgpu-ffi\",\"version\":\"unavailable\",\"ffi_api_version\":" +
           std::to_string(kFfiApiVersion) + ",\"error\":\"" + json_escape(message) + "\"}";
}

const ZkgpuFfi& ffi() {
    static ZkgpuFfi resolved;
    static std::once_flag once;
    std::call_once(once, []() {
        resolved.handle = dlopen(kRustLibrary, RTLD_NOW | RTLD_LOCAL);
        if (resolved.handle == nullptr) {
            const char* dl_err = dlerror();
            resolved.error = std::string("unable to load ") + kRustLibrary + ": " +
                             (dl_err == nullptr ? "unknown dynamic loader error" : dl_err);
            __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", resolved.error.c_str());
            return;
        }

        resolved.run_request_json = reinterpret_cast<RunRequestJsonFn>(
                dlsym(resolved.handle, "zkgpu_run_request_json"));
        resolved.get_version_json = reinterpret_cast<GetVersionJsonFn>(
                dlsym(resolved.handle, "zkgpu_get_version_json"));
        resolved.free_string = reinterpret_cast<FreeStringFn>(
                dlsym(resolved.handle, "zkgpu_free_string"));

        if (resolved.run_request_json == nullptr ||
            resolved.get_version_json == nullptr ||
            resolved.free_string == nullptr) {
            resolved.error = "missing required zkgpu_ffi symbols";
            __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", resolved.error.c_str());
        }
    });
    return resolved;
}

std::string jstring_to_utf8(JNIEnv* env, jstring text) {
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID get_bytes = env->GetMethodID(string_class, "getBytes", "(Ljava/lang/String;)[B");
    jstring utf8 = env->NewStringUTF("UTF-8");
    auto* bytes = static_cast<jbyteArray>(env->CallObjectMethod(text, get_bytes, utf8));

    const jsize length = env->GetArrayLength(bytes);
    std::string out(static_cast<size_t>(length), '\0');
    env->GetByteArrayRegion(bytes, 0, length, reinterpret_cast<jbyte*>(out.data()));

    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(utf8);
    env->DeleteLocalRef(string_class);
    return out;
}

jstring utf8_to_jstring(JNIEnv* env, const std::string& text) {
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    auto* bytes = env->NewByteArray(static_cast<jsize>(text.size()));
    env->SetByteArrayRegion(
            bytes, 0, static_cast<jsize>(text.size()),
            reinterpret_cast<const jbyte*>(text.data()));
    jstring utf8 = env->NewStringUTF("UTF-8");
    auto* result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, utf8));

    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(utf8);
    env->DeleteLocalRef(string_class);
    return result;
}

std::string call_run_request(const std::string& request_json) {
    const auto& api = ffi();
    if (api.run_request_json == nullptr || api.free_string == nullptr) {
        return error_json(api.error.empty() ? "zkgpu_ffi is unavailable" : api.error);
    }

    char* response = api.run_request_json(request_json.c_str());
    if (response == nullptr) {
        return error_json("zkgpu_run_request_json returned null");
    }

    std::string json(response);
    api.free_string(response);
    return json;
}

std::string call_get_version() {
    const auto& api = ffi();
    if (api.get_version_json == nullptr || api.free_string == nullptr) {
        return version_error_json(api.error.empty() ? "zkgpu_ffi is unavailable" : api.error);
    }

    char* response = api.get_version_json();
    if (response == nullptr) {
        return version_error_json("zkgpu_get_version_json returned null");
    }

    std::string json(response);
    api.free_string(response);
    return json;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_org_zkgpu_harness_ZkgpuBridge_runRequestJson(JNIEnv* env, jclass, jstring request_json) {
    if (request_json == nullptr) {
        return utf8_to_jstring(env, error_json("requestJson must not be null"));
    }
    return utf8_to_jstring(env, call_run_request(jstring_to_utf8(env, request_json)));
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_zkgpu_harness_ZkgpuBridge_getVersionJson(JNIEnv* env, jclass) {
    return utf8_to_jstring(env, call_get_version());
}
