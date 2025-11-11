git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/bigdata/appointments.csv \
   data/bigdata/billing.csv \
   data/bigdata/patients.csv \
   data/bigdata/treatments.csv" \
  --prune-empty --tag-name-filter cat -- --all

