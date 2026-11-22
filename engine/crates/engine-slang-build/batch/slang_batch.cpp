// slang_batch: compiles several entry points of one Slang source in one process, so the
// source and its includes are parsed and checked once instead of once per entry (slangc
// takes about half a second per entry for that alone on the rigid kernels).
//
// Used by engine-slang-build (`compile_many`) only where its output is byte for byte
// slangc's: same Slang build (checked with --version against slangc), same relative
// paths and working directory, one entry per linked program. Anything it does not write
// (an error, an entry it skips, a run cut off by the caller's time limit) is compiled
// by slangc as before. engine/tools/build-time/kernel_digest.py checks the outputs.
//
//   slang_batch --version          prints the Slang build tag
//   slang_batch JOB                JOB lines:
//     target wgsl|spirv|metal|cuda
//     source <path>                (relative to the working directory, as slangc's)
//     include <dir>                (repeatable)
//     define <NAME>=<VALUE>        (repeatable)
//     entry <name> <stage> <out>   (repeatable)
//   Prints "ok <index>" after each entry's output is in place (written beside it and
//   renamed), "fail <index>" when an entry did not compile.
#include <slang.h>
#include <slang-com-ptr.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using Slang::ComPtr;

static SlangStage stage_of(const std::string& s) {
    if (s == "compute") return SLANG_STAGE_COMPUTE;
    if (s == "vertex") return SLANG_STAGE_VERTEX;
    if (s == "fragment") return SLANG_STAGE_FRAGMENT;
    return SLANG_STAGE_NONE;
}

static SlangCompileTarget target_of(const std::string& t) {
    if (t == "wgsl") return SLANG_WGSL;
    if (t == "spirv") return SLANG_SPIRV;
    if (t == "metal") return SLANG_METAL;
    if (t == "cuda") return SLANG_CUDA_SOURCE;
    return SLANG_TARGET_UNKNOWN;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        ComPtr<slang::IGlobalSession> global;
        if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) return 1;
        printf("%s\n", global->getBuildTagString());
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: slang_batch --version | slang_batch JOB\n");
        return 2;
    }
    std::ifstream in(argv[1]);
    std::string line, target, source;
    std::vector<std::string> includes, names, values;
    struct Entry { std::string name, stage, out; };
    std::vector<Entry> entries;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string k;
        ls >> k;
        if (k == "target") ls >> target;
        else if (k == "source") ls >> source;
        else if (k == "include") { std::string d; ls >> d; includes.push_back(d); }
        else if (k == "define") {
            std::string d;
            ls >> d;
            auto eq = d.find('=');
            names.push_back(d.substr(0, eq));
            values.push_back(eq == std::string::npos ? "" : d.substr(eq + 1));
        } else if (k == "entry") { Entry e; ls >> e.name >> e.stage >> e.out; entries.push_back(e); }
    }
    if (target_of(target) == SLANG_TARGET_UNKNOWN || source.empty()) {
        fprintf(stderr, "slang_batch: bad job\n");
        return 2;
    }

    ComPtr<slang::IGlobalSession> global;
    if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) return 1;
    slang::TargetDesc td{};
    td.format = target_of(target);
    std::vector<slang::PreprocessorMacroDesc> macros;
    for (size_t i = 0; i < names.size(); i++) macros.push_back({names[i].c_str(), values[i].c_str()});
    std::vector<const char*> paths;
    for (auto& d : includes) paths.push_back(d.c_str());
    slang::SessionDesc sd{};
    sd.targets = &td;
    sd.targetCount = 1;
    sd.searchPaths = paths.data();
    sd.searchPathCount = (SlangInt)paths.size();
    sd.preprocessorMacros = macros.data();
    sd.preprocessorMacroCount = (SlangInt)macros.size();
    // slangc's default (the session's own default is row major).
    sd.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
    ComPtr<slang::ISession> session;
    if (SLANG_FAILED(global->createSession(sd, session.writeRef()))) return 1;

    std::ifstream src(source);
    std::stringstream text;
    text << src.rdbuf();
    std::string code_text = text.str();
    ComPtr<slang::IBlob> diag;
    slang::IModule* module =
        session->loadModuleFromSourceString("slang_batch_module", source.c_str(), code_text.c_str(), diag.writeRef());
    if (!module) {
        fprintf(stderr, "%s\n", diag ? (const char*)diag->getBufferPointer() : "slang_batch: load failed");
        return 1;
    }
    for (size_t i = 0; i < entries.size(); i++) {
        const Entry& e = entries[i];
        ComPtr<slang::IEntryPoint> ep;
        ComPtr<slang::IBlob> d;
        ComPtr<slang::IComponentType> composite, linked;
        ComPtr<slang::IBlob> code;
        bool ok = SLANG_SUCCEEDED(module->findAndCheckEntryPoint(e.name.c_str(), stage_of(e.stage), ep.writeRef(), d.writeRef()));
        if (ok) {
            slang::IComponentType* parts[] = {module, ep.get()};
            ok = SLANG_SUCCEEDED(session->createCompositeComponentType(parts, 2, composite.writeRef(), d.writeRef()))
                && SLANG_SUCCEEDED(composite->link(linked.writeRef(), d.writeRef()))
                && SLANG_SUCCEEDED(linked->getEntryPointCode(0, 0, code.writeRef(), d.writeRef()));
        }
        if (ok) {
            std::string tmp = e.out + ".batch-tmp";
            FILE* f = fopen(tmp.c_str(), "wb");
            ok = f && fwrite(code->getBufferPointer(), 1, code->getBufferSize(), f) == code->getBufferSize();
            if (f) fclose(f);
            ok = ok && std::rename(tmp.c_str(), e.out.c_str()) == 0;
        }
        printf("%s %zu\n", ok ? "ok" : "fail", i);
        fflush(stdout);
    }
    return 0;
}
