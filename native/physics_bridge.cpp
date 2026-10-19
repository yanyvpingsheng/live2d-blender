// Original, headless C ABI adapter. Build against a separately installed,
// appropriately licensed Cubism Native SDK 5-r.5. No SDK source is vendored.
// The resulting library contains Framework code and is NOT a redistributable
// part of the Python add-on package without a separate license review.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "CubismFramework.hpp"
#include "ICubismAllocator.hpp"
#include "Model/CubismMoc.hpp"
#include "Model/CubismModel.hpp"
#include "Physics/CubismPhysics.hpp"
#include "Effect/CubismPose.hpp"
#include "Id/CubismIdManager.hpp"
#include "Rendering/CubismRenderer.hpp"

#if defined(_WIN32)
#define L2DP_EXPORT extern "C" __declspec(dllexport)
#else
#define L2DP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace fw = Live2D::Cubism::Framework;
namespace core = Live2D::Cubism::Core;

// Framework disposal expects a platform renderer cleanup entry point. This
// headless library never constructs a renderer or allocates graphics resources.
namespace Live2D { namespace Cubism { namespace Framework { namespace Rendering {
void CubismRenderer::StaticRelease() {}
}}}}

namespace {
class Allocator final : public fw::ICubismAllocator {
public:
    void* Allocate(fw::csmSizeType size) override { return std::malloc(size); }
    void Deallocate(void* address) override { std::free(address); }
    void* AllocateAligned(fw::csmSizeType size, fw::csmUint32 alignment) override {
        if (!alignment || (alignment & (alignment - 1))) return nullptr;
        const size_t extra = alignment - 1 + sizeof(void*);
        if (size > SIZE_MAX - extra) return nullptr;
        void* base = std::malloc(size + extra);
        if (!base) return nullptr;
        const auto begin = reinterpret_cast<uintptr_t>(base) + sizeof(void*);
        const auto aligned = (begin + alignment - 1) & ~(uintptr_t(alignment) - 1);
        reinterpret_cast<void**>(aligned)[-1] = base;
        return reinterpret_cast<void*>(aligned);
    }
    void DeallocateAligned(void* address) override {
        if (address) std::free(reinterpret_cast<void**>(address)[-1]);
    }
};

Allocator allocator;
std::mutex apiMutex;
unsigned liveHandles = 0;
thread_local std::string errorMessage;

void initializeFramework() {
    if (!liveHandles) {
        if (!fw::CubismFramework::StartUp(&allocator))
            throw std::runtime_error("Cubism Framework startup failed");
        fw::CubismFramework::Initialize();
    }
    ++liveHandles;
}

void releaseFramework() {
    if (liveHandles && --liveHandles == 0) {
        fw::CubismFramework::Dispose();
        fw::CubismFramework::CleanUp();
    }
}

struct Handle {
    fw::CubismMoc* moc = nullptr;
    fw::CubismModel* model = nullptr;
    fw::CubismPhysics* physics = nullptr;
    fw::CubismPose* pose = nullptr;
    std::vector<unsigned char> physicsJson;
    std::vector<unsigned char> poseJson;

    void clearRuntime() {
        if (physics) fw::CubismPhysics::Delete(physics);
        if (pose) fw::CubismPose::Delete(pose);
        if (model && moc) moc->DeleteModel(model);
        physics = nullptr; pose = nullptr; model = nullptr;
    }
    ~Handle() {
        clearRuntime();
        if (moc) fw::CubismMoc::Delete(moc);
    }
    void recreate() {
        clearRuntime();
        model = moc->CreateModel();
        if (!model) throw std::runtime_error("Cubism model initialization failed");
        if (!physicsJson.empty()) {
            physics = fw::CubismPhysics::Create(physicsJson.data(), int(physicsJson.size()));
            if (!physics) throw std::runtime_error("Cubism rejected physics3 JSON");
        }
        if (!poseJson.empty()) {
            pose = fw::CubismPose::Create(poseJson.data(), int(poseJson.size()));
            if (!pose) throw std::runtime_error("Cubism rejected pose3 JSON");
            // Initialize the SDK's virtual pose parameters and remember model.
            // Caller input is applied after this initial state is established.
            pose->UpdateParameters(model, 0.0f);
        }
    }
};

Handle& handle(void* pointer) {
    if (!pointer) throw std::invalid_argument("Physics bridge handle is closed");
    Handle& value = *static_cast<Handle*>(pointer);
    if (!value.model) throw std::runtime_error("Physics bridge needs reset after failure");
    return value;
}

void finite(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("Non-finite input");
}

template<typename Operation> int status(Operation operation) {
    std::lock_guard<std::mutex> lock(apiMutex);
    errorMessage.clear();
    try { operation(); return 1; }
    catch (const std::exception& error) { errorMessage = error.what(); }
    catch (...) { errorMessage = "Unknown Cubism bridge failure"; }
    return 0;
}
}

L2DP_EXPORT unsigned l2dp_abi_version() { return 1; }
L2DP_EXPORT unsigned l2dp_core_version() { return core::csmGetVersion(); }
L2DP_EXPORT const char* l2dp_last_error() { return errorMessage.c_str(); }

L2DP_EXPORT void* l2dp_create(const void* moc, unsigned mocSize,
                            const void* physics, unsigned physicsSize,
                            const void* pose, unsigned poseSize) {
    std::lock_guard<std::mutex> lock(apiMutex);
    errorMessage.clear();
    bool initialized = false;
    try {
        if (!moc || mocSize < 4 || mocSize > 512U * 1024 * 1024 ||
            physicsSize > 32U * 1024 * 1024 || poseSize > 32U * 1024 * 1024 ||
            (physicsSize && !physics) || (poseSize && !pose))
            throw std::invalid_argument("Invalid input buffer or file size");
        initializeFramework(); initialized = true;
        std::unique_ptr<Handle> result(new Handle);
        result->moc = fw::CubismMoc::Create(static_cast<const unsigned char*>(moc), int(mocSize), true);
        if (!result->moc) throw std::runtime_error("Cubism rejected MOC consistency/version");
        if (physicsSize) result->physicsJson.assign(static_cast<const unsigned char*>(physics), static_cast<const unsigned char*>(physics) + physicsSize);
        if (poseSize) result->poseJson.assign(static_cast<const unsigned char*>(pose), static_cast<const unsigned char*>(pose) + poseSize);
        result->recreate();
        return result.release();
    } catch (const std::exception& error) { errorMessage = error.what(); }
      catch (...) { errorMessage = "Unknown Cubism creation failure"; }
    if (initialized) releaseFramework();
    return nullptr;
}

L2DP_EXPORT void l2dp_destroy(void* pointer) {
    std::lock_guard<std::mutex> lock(apiMutex);
    if (pointer) { delete static_cast<Handle*>(pointer); releaseFramework(); }
}

L2DP_EXPORT int l2dp_reset(void* pointer) {
    // Recreate instead of Physics::Reset: remove all time/input/output caches
    // and virtual parameters, so rewinding really matches a fresh instance.
    return status([&] { if (!pointer) throw std::invalid_argument("Closed bridge");
                       static_cast<Handle*>(pointer)->recreate(); });
}

L2DP_EXPORT int l2dp_parameter_count(void* pointer) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return pointer ? static_cast<Handle*>(pointer)->model->GetParameterCount() : -1;
}
L2DP_EXPORT int l2dp_part_count(void* pointer) {
    std::lock_guard<std::mutex> lock(apiMutex);
    return pointer ? static_cast<Handle*>(pointer)->model->GetPartCount() : -1;
}
L2DP_EXPORT const char* l2dp_parameter_id(void* pointer, int index) {
    if (!pointer) return nullptr;
    auto model = static_cast<Handle*>(pointer)->model;
    return index >= 0 && index < model->GetParameterCount() ? model->GetParameterId(index)->GetString().GetRawString() : nullptr;
}
L2DP_EXPORT const char* l2dp_part_id(void* pointer, int index) {
    if (!pointer) return nullptr;
    auto model = static_cast<Handle*>(pointer)->model;
    return index >= 0 && index < model->GetPartCount() ? model->GetPartId(index)->GetString().GetRawString() : nullptr;
}
L2DP_EXPORT int l2dp_defaults(void* pointer, float* values, int count) {
    return status([&] {
        auto model = handle(pointer).model;
        if (count != model->GetParameterCount() || (count && !values)) throw std::invalid_argument("Parameter buffer size mismatch");
        for (int i = 0; i < count; ++i) values[i] = model->GetParameterDefaultValue(i);
    });
}
L2DP_EXPORT int l2dp_set_parameters(void* pointer, const float* values, int count) {
    return status([&] {
        auto model = handle(pointer).model;
        if (count != model->GetParameterCount() || (count && !values)) throw std::invalid_argument("Parameter buffer size mismatch");
        for (int i = 0; i < count; ++i) finite(values[i]);
        for (int i = 0; i < count; ++i) model->SetParameterValue(i, values[i]);
    });
}
L2DP_EXPORT int l2dp_set_parameter_id(void* pointer, const char* id, float value) {
    return status([&] {
        finite(value);
        if (!id || !*id) throw std::invalid_argument("Empty parameter ID");
        handle(pointer).model->SetParameterValue(fw::CubismFramework::GetIdManager()->GetId(id), value);
    });
}
L2DP_EXPORT int l2dp_set_part(void* pointer, int index, float value) {
    return status([&] {
        auto model = handle(pointer).model; finite(value);
        if (index < 0 || index >= model->GetPartCount() || value < 0 || value > 1)
            throw std::invalid_argument("Invalid part index/opacity");
        model->SetPartOpacity(index, value);
    });
}
L2DP_EXPORT int l2dp_stabilize(void* pointer) {
    return status([&] { auto& h = handle(pointer); if (h.physics) h.physics->Stabilization(h.model); });
}
L2DP_EXPORT int l2dp_step(void* pointer, float delta, int phase) {
    return status([&] {
        auto& h = handle(pointer); finite(delta);
        if (delta < 0 || delta > 5 || phase < 0 || phase > 2)
            throw std::invalid_argument("Delta outside 0..5 seconds or invalid phase; replay seeks in small steps");
        // R5 late-update order: physics (600), lipsync (700), pose (800).
        // phase 0 = physics+pose; 1/2 allow caller-provided lipsync in between.
        if (phase != 2 && h.physics) h.physics->Evaluate(h.model, delta);
        if (phase != 1 && h.pose) h.pose->UpdateParameters(h.model, delta);
    });
}
L2DP_EXPORT int l2dp_read(void* pointer, float* parameters, int parameterCount,
                          float* parts, int partCount) {
    return status([&] {
        auto model = handle(pointer).model;
        if (parameterCount != model->GetParameterCount() || partCount != model->GetPartCount() ||
            (parameterCount && !parameters) || (partCount && !parts))
            throw std::invalid_argument("Read buffer size mismatch");
        for (int i = 0; i < parameterCount; ++i) {
            parameters[i] = model->GetParameterValue(i); finite(parameters[i]);
        }
        for (int i = 0; i < partCount; ++i) {
            parts[i] = model->GetPartOpacity(i); finite(parts[i]);
        }
    });
}
