# %%
if __name__ == '__main__':


    add_row_from_excel("/home/ub/code/fran/configurations/experiment_configs_liver.xlsx",sheet_name="plan2")
    add_row_from_excel("/home/ub/code/fran/configurations/experiment_configs_liver.xlsx",sheet_name="plan4")
    db_path="plans.db"
    _init_db(db_path)
# Example:
# new_id = add_row_from_excel("path/to/table.xlsx")
# print("row id:", new_id)

