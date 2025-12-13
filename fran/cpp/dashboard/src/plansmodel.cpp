#include "plansmodel.h"
#include <QBrush>
#include <QColor>
#include <qnamespace.h>

int PlansModel::rowCount(const QModelIndex &) const {
  size_t n = m_plans.rowCount();
  int n0 = static_cast<int>(n);
  return n0;
}
int PlansModel::columnCount(const QModelIndex &) const {
  size_t n = m_plans.columnCount();
  int n0 = static_cast<int>(n);
  return n0;
}

QVariant PlansModel::headerData(int section, Qt::Orientation orientation,
                                int role) const {

  if (role != Qt::DisplayRole || orientation != Qt::Horizontal) {
    return QVariant();
  }
  std::vector<std::string> header = m_plans.header();
  std::string val = header[section];
  QString val1 = QString::fromStdString(val);
  return val1;
}

QVariant PlansModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid()) {
    return QVariant();
  }
  int col = index.column();
  int row = index.row();
  if (col == 0) {
    if (role == Qt::DecorationRole){
      QColor r = preproccedStatus(row);
      return r;
    }
  }
  if (role == Qt::DisplayRole) {
    std::vector<std::string> fullRow = m_plans[index.row()];
    std::string val = fullRow[index.column()];
    QString val1 = QString::fromStdString(val);
    return val1;
  } else {
    return QVariant();
  };
}

QColor PlansModel::preproccedStatus (int index) const {
  std::string val = m_preprocessed[index];
  QColor color;
  if (val=="both"){
    color = QColor(0, 255, 0);
  }
  else if (val=="one") {
    color = QColor(255, 255,0);
  }
  else {
    color = QColor(255, 0, 0);
  }

    return color;
}

