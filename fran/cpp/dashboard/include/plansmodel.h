#pragma once
#include "franproject.h"
#include <QAbstractTableModel>
#include <QStyledItemDelegate>
#include <iostream>

class PlansModel : public QAbstractTableModel {
  Q_OBJECT
public:
  explicit PlansModel(PlansDF &plansDF, QObject *parent = nullptr)
      : QAbstractTableModel(parent), m_plans(plansDF) {

    size_t n = m_plans.rowCount();
    for (size_t i = 0; i < n; i++) {
      std::string done = "both";
      m_preprocessed.push_back(done);
    }
    std::vector<std::string> header = m_plans.header();
    for (int i = 0; i < header.size(); i++) {
      std::cout << "==========================================";

      std::cout << header[i] << std::endl;
    }

    m_preprocessed = m_plans["preprocessed"];
  }

  int columnCount(const QModelIndex &) const override;
  int rowCount(const QModelIndex &) const override;
  QVariant data(const QModelIndex &index, int role) const override;
  QVariant headerData(int section, Qt::Orientation orientation,
                      int role) const override;
  QColor preproccedStatus(int index) const;

private:
  const PlansDF &m_plans;
  std::vector<std::string> m_preprocessed;
};




class AnalyzeButtonDelegate : public QStyledItemDelegate {
  Q_OBJECT
public:
  explicit AnalyzeButtonDelegate(QObject *parent = nullptr,
                                 const int num_columns = 0);

signals:
  void clickedRow(const int row);
  void clicked(const QModelIndex &index);

protected:
  void paint(QPainter *painter, const QStyleOptionViewItem &option,
             const QModelIndex &index) const override;
  bool editorEvent(QEvent *event, QAbstractItemModel *model,
                   const QStyleOptionViewItem &option,
                   const QModelIndex &index) override;

private:
  int m_columns;
};
