#include "python_env.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

PythonEnv &PythonEnv::instance() {
    static PythonEnv env;
    return env;
}

PythonEnv::PythonEnv() {
    auto path = fs::path("/proc/self/exe");
    auto exe = std::filesystem::canonical(path);
    auto build_dir = exe.parent_path();
    auto build_dir2 = build_dir.parent_path();
    auto proj_dir  = build_dir2.parent_path();
    auto cpp_dir = proj_dir.parent_path();
    auto fran_dir = cpp_dir.parent_path();

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, fran_dir.string());
    std::cout<< fran_dir.string() << std::endl;

    m_app_ = py::module_::import("fran.managers.project");
}

