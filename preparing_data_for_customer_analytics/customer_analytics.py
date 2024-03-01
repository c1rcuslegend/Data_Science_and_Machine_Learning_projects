import pandas as pd

ds_jobs = pd.read_csv("./customer_train.csv")

ordered_categories={
        "relevant_experience":['No relevant experience','Has relevant experience'],
        "enrolled_university":['no_enrollment', 'Part time course','Full time course'],
        "education_level":['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
        "experience":["<1"] + [str(i) for i in range(1,21)]+[">20"],
        "company_size":["<10","10-49","50-99","100-499","500-999","1000-4999","5000-9999","10000+"],
        "last_new_job":["never","1","2","3","4",">4"]
    }

new_cats={}
for col, value in ds_jobs.dtypes.items():
    if value=="int64":
        new_cats[col]="int32"
    elif value=="float64":
        new_cats[col]="float16"
    elif col in ordered_categories:
        new_cats[col] = pd.CategoricalDtype(ordered_categories[col], ordered=True)
    else:
        new_cats[col] = "category"

ds_jobs_clean = ds_jobs.copy()
ds_jobs_clean = ds_jobs.astype(new_cats)

print(ds_jobs.memory_usage())
print(ds_jobs_clean.memory_usage())

ds_jobs_clean = ds_jobs_clean[(ds_jobs_clean["experience"]>="10") & (ds_jobs_clean["company_size"]>="1000-4999")]
