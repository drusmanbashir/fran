#include "franproject.h"
#include <filesystem>
#include <iostream>
#include <pybind11/pytypes.h>
#include <xlnt/xlnt.hpp>

namespace fs = std::filesystem;

FranProject &FranProject::instance() {
  static FranProject env;
  return env;
}

FranProject::FranProject() {
  auto path = fs::path("/proc/self/exe");
  auto exe = std::filesystem::canonical(path);
  auto build_dir = exe.parent_path();
  auto build_dir2 = build_dir.parent_path();
  auto proj_dir = build_dir2.parent_path();
  auto cpp_dir = proj_dir.parent_path();
  auto fran_dir = cpp_dir.parent_path();

  py::module_ sys = py::module_::import("sys");
  sys.attr("path").attr("insert")(0, fran_dir.string());
  std::cout << fran_dir.string() << std::endl;
  m_proj_mod = py::module_::import("fran.managers.project");
  m_analyze_resample = py::module_::import("fran.run.analyze_resample");
}

void FranProject::loadProject(std::string project_name, std::string mnemonic) {
  py::gil_scoped_acquire gil;
  py::object proj_obj = m_proj_mod.attr("Project")(project_name);
  m_global_properties = proj_obj.attr("global_properties");
  m_project_title = m_global_properties["project_title"].cast<std::string>();
  // QMessageBox::information(nullptr, "Project loaded", title.c_str());
  py::module conf_mod = py::module_::import("fran.configs.parser");
  py::object conf_obj = conf_mod.attr("ConfigMaker")(proj_obj);
  conf_obj.attr("add_preprocess_status")();
  m_plansDF = PlansDF(conf_obj.attr("plans"));
  // py::object Conf = conf_obj(proj_obj);
}

std::string FranProject::parseDict() {
  py::gil_scoped_acquire gil;
  std::string s = py::str(m_global_properties);
  return s;
}

bool FranProject::plan_loaded() {
  if (!m_plansDF) {
    return false;
  }
  return true;
}

std::vector<std::string> PlansDF::operator[](size_t i) const {

  py::object iloc = m_plans_py.attr("iloc");
  py::object row = iloc[py::int_(i)];
  std::vector<std::string> row_vector;
  for (auto c : row) {
    row_vector.push_back(py::str(c));
  };
  return row_vector;
}

std::vector<std::string> PlansDF::operator[](std::string colname) const {
  py::gil_scoped_acquire gil;
  py::object col = m_plans_py[colname.c_str()];
  std::vector<std::string> colVector;
  for (auto c : col) {
    colVector.push_back(py::str(c));
  }
  return colVector;
}

std::vector<std::string> PlansDF::header() const {
  py::gil_scoped_acquire gil;
  py::object cols = m_plans_py.attr("columns");
  std::vector<std::string> header;
  for (auto c : cols) {
    std::string colname = py::str(c);

    header.push_back(py::str(c));
  };

  return header;
}

bool PlansDF::has_column(const std::string &name) const {
  if (m_plans_py.is_none())
    return false;
  py::gil_scoped_acquire gil;
  py::object cols = m_plans_py.attr("columns");
  return py::bool_(cols.attr("__contains__")(name)).cast<bool>();
}

size_t PlansDF::rowCount() const {

  py::gil_scoped_acquire gil;
  py::tuple shape = m_plans_py.attr("shape").cast<py::tuple>();
  return shape[0].cast<size_t>();
}

size_t PlansDF::columnCount() const {
  py::gil_scoped_acquire gil;
  py::tuple shape = m_plans_py.attr("shape").cast<py::tuple>();
  return shape[1].cast<size_t>();
}

py::object PlansDF::operator()(size_t i, std::string colname) const {
  py::gil_scoped_acquire gil;
  py::object column = m_plans_py.attr("__getitem__")(colname.c_str());
  py::object item = column.attr("iloc")[py::int_(i)];
  return item;
}

py::object PlansDF::operator()(size_t i, size_t j) const {
  return m_plans_py.attr("iloc")[py::make_tuple(i, j)];
}

void FranProject::run_analyze_resample(int plan_id, int n_procs,
                                       bool overwrite) {
  py::gil_scoped_acquire gil;
  py::module_ argparse = py::module_::import("argparse");
  py::object js = argparse.attr("Namespace")(
      py::arg("project_title") = m_project_title, py::arg("plan") = plan_id,
      py::arg("num_processes") = n_procs, py::arg("overwrite") = overwrite);

  m_analyze_resample.attr("main")(js);
}
