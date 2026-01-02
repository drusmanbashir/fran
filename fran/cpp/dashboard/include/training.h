#pragma once
#include "franproject.h"
#include <QString>
#include <QVariantMap>

struct TrainingArgs {
  int plan_id;
  std::vector<int> devices;
  double lr;
  int epochs;
  int bs;
  int k_largest;
  int fold;
  bool overwrite;
  bool compiled;
};
