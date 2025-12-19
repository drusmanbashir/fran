#include "plansmodel.h"
#include <QBrush>
#include <QColor>
#include <QEvent>
#include <QMouseEvent>
#include <qapplication.h>
#include <qcoreevent.h>
#include <qstyleoption.h>

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
    if (role == Qt::DecorationRole) {
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

QColor PlansModel::preproccedStatus(int index) const {
  std::string val = m_preprocessed[index];
  QColor color;
  if (val == "both") {
    color = QColor(0, 255, 0);
  } else if (val == "one") {
    color = QColor(255, 255, 0);
  } else {
    color = QColor(255, 0, 0);
  }
  return color;
}

AnalyzeButtonDelegate::AnalyzeButtonDelegate(QObject *parent,
                                             const int num_columns)
    : QStyledItemDelegate(parent), m_columns(num_columns) {};

void AnalyzeButtonDelegate::paint(QPainter *painter,
                                  const QStyleOptionViewItem &option,
                                  const QModelIndex &index) const {
  QStyleOptionButton btn;
  btn.rect = option.rect.adjusted(4, 2, -4, -2);
  btn.state = QStyle::State_Enabled | QStyle::State_Raised;
  btn.text = "Analyze";
  if (option.state & QStyle::State_MouseOver) {
    btn.state = btn.state | QStyle::State_MouseOver;
  }
  if (option.state & QStyle::State_Sunken) {
    btn.state &= ~QStyle::State_Raised;
    btn.features = QStyleOptionButton::Flat;
  }
  QApplication::style()->drawControl(QStyle::CE_PushButton, &btn, painter);
}

bool AnalyzeButtonDelegate::editorEvent(QEvent *event,
                                        QAbstractItemModel *model,
                                        const QStyleOptionViewItem &option,
                                        const QModelIndex &index) {
  if (event->type() == QEvent::MouseButtonRelease) {
    auto *me = static_cast<QMouseEvent *>(event);
    const QRect r = option.rect.adjusted(4, 2, -4, -2);
    if (r.contains(me->pos())) {
      emit clicked(index);
      emit clickedRow(index.row());
      return true;
    }
  }
  return false;
}

// explicit AnalyzeButtonDelegate::AnalyzeButtonDelegate (QObject *parent) :
// QStyledItemDelegate (parent) {
//
// };
