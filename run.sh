

# for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn"
for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn"
do
    echo ${d_name}
    # python -u main.py --identifier 082319_datasets --model_name spline ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 --d_name ${d_name} &> logs/082319_datasets_${d_name}.log &
    python -u main.py --identifier 082319_datasets --model_name ebm-bf ebm-bf-o8 --d_name ${d_name} &> logs/082319_datasets_ebm_bf_${d_name}.log &
done

# spline not finished in credit and mimic_iii


# Why the xgb now cutting in asthma!? Run another xgb with the min_child_weight as 0 called xgb-d1-cw0
for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn"
do
    echo ${d_name}
    # python -u main.py --identifier 082319_datasets --model_name spline ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 --d_name ${d_name} &> logs/082319_datasets_${d_name}.log &
    python -u main.py --identifier 082319_datasets --model_name xgb-d1-cw0 --d_name ${d_name} &> logs/082319_datasets_xgb-d1-cw0_${d_name}.log &
done


# bias variance tradeoff....
for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn" "adult" "credit" "mushroom"
do
    echo ${d_name}
    python -u main.py --identifier 082419_bias_var --exp_mode bias_var --model_name ebm skgbt xgb-d1 xgb-d1-o12 ebm-o12 skgbt-o12 ebm-bf ebm-bf-o12 spline --d_name ${d_name} &> logs/082419_bias_var_${d_name}.log &
done

python -u main.py --identifier debug4 --model_name spline --d_name mimic_iii
python -u main.py --identifier debug6 --model_name spline-v --d_name mimic_iii --exp_mode bias_var

## Running this
for dset_name in "pneumonia" "mimiciii"
do
    for b in "0" "1"
    do
        for d_name in "ss_${dset_name}_b${b}_r1377_xgb-d1-o16" "ss_${dset_name}_b${b}_r1377_ebm-o16-i16" "ss_${dset_name}_b${b}_r1377_ebm-bf-o16-i16"
        do
            echo ${d_name}
            python -u main.py --identifier 082419_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/082419_semi_synthetic_${d_name}.log &
        done
    done
done

python -u main.py --identifier debug8 --model_name spline --d_name ss_pneumonia_b0_r1377_xgb-d1

## Running spline
for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn" "adult" "credit" "mushroom"
do
    echo ${d_name}
    python -u main.py --identifier 082319_datasets --model_name spline --d_name ${d_name} --overwrite 1 &> logs/082519_datasets_spline_${d_name}.log &
done \
&& \
for d_name in "pneumonia" "mimic_iii" "breast" "bikeshare" "churn" "adult" "credit" "mushroom" ## Run bias var
do
    echo ${d_name}
    python -u main.py --identifier 082419_bias_var --exp_mode bias_var --model_name ebm skgbt xgb-d1 xgb-d1-o12 ebm-o12 skgbt-o12 ebm-bf ebm-bf-o12 spline --d_name ${d_name} &> logs/082419_bias_var_${d_name}.log &
done
&& \
for d_name in "adult" "credit" "mushroom"  ## Running the rest of the dataset?
do
    echo ${d_name}
    python -u main.py --identifier 082319_datasets --model_name ebm skgbt xgb-d1 xgb-d1-o12 ebm-o12 skgbt-o12 ebm-bf ebm-bf-o12 spline --d_name ${d_name} &> logs/082419_datasets_${d_name}.log &
done \


## mushroom and mimiciii spline still dies Orz... Dies in the re-gridsearch stage
## I accidentally cancel a job of ss. So we might lose a dataset (mimic0?)
## vim logs/082519_datasets_spline_mimiciii.log


# Wait and see
for d_name in "mimiciii"
do
    for m_name in ebm skgbt xgb-d1 xgb-d1-o12 ebm-o12 skgbt-o12 ebm-bf ebm-bf-o12 spline
    do
        echo ${d_name} ${m_name}
        python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name ${d_name} --overwrite 1 &> logs/082719_${d_name}_${m_name}.log &
    done
done


# Bias Variance tradeoff rerun
for d_name in "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "credit" "mushroom"
do
    echo ${d_name}
    python -u main.py --identifier 082419_bias_var --exp_mode bias_var --model_name ebm skgbt xgb-d1 xgb-d1-o12 ebm-o12 skgbt-o12 ebm-bf ebm-bf-o12 spline --d_name ${d_name} &> logs/082619_bias_var_${d_name}.log &
done


for d_name in "mushroom"
do
    for m_name in spline
    do
        echo ${d_name} ${m_name}
        python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name ${d_name} &> logs/082719_${d_name}_${m_name}.log &
    done
done

# SKGBT since we upgrad hey can't be used anymore Orz....
for m_name in skgbt
do
    echo ${m_name}
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name pneumonia mimiciii breast bikeshare churn adult credit mushroom &> logs/082819_rerun_skgbt_all.log &
done

# Run a SS for all
for dset_name in "pneumonia" "mimiciii"
do
    for b in "0" "1"
    do
        for d_name in "ss_${dset_name}_b${b}_r1377_xgb-d1-o16" "ss_${dset_name}_b${b}_r1377_ebm-o16-i16" "ss_${dset_name}_b${b}_r1377_ebm-bf-o16-i16"
        do
            echo ${d_name}
            python -u main.py --identifier 082419_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/082419_semi_synthetic_${d_name}.log &
        done
    done
done


## Needs to run a more bagging for xgb to compensate its performance....?
for m_name in xgb-d1-o16
do
    echo ${m_name}
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name pneumonia mimiciii breast bikeshare churn adult credit mushroom &> logs/082819_rerun_${m_name}_all.log &
done


python -u main.py --identifier debug12 --model_name xgb-d1 --d_name pneumonia


# Regenerate all the efforts
for d_name in "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "mushroom"; do
python -u main.py --identifier 082319_datasets --model_name ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 ebm-bf ebm-bf-o8 spline --d_name ${d_name} &> logs/082819_regenerate_${d_name}.log &
done

for d_name in "credit"; do
python -u main.py --identifier 082319_datasets --model_name ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 ebm-bf ebm-bf-o8 spline --d_name ${d_name} &> logs/082819_regenerate_${d_name}.log &
done

# After that, run the credit dataset in parallel...
for m_name in ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 ebm-bf ebm-bf-o8 spline; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name credit &> logs/082819_regenerate_credit.log &
done

# After that, rerun the bias var tradeoff and cache the model....
for d_name in "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "credit" "mushroom" ## Run bias var
do
    echo ${d_name}
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name lr ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/082819_bias_var_${d_name}.log &
done

# Run the LR
for m_name in lr mlr ilr; do
python -u main.py --identifier 082319_datasets --overwrite 1 --model_name ${m_name} --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "mushroom" "credit" &> logs/082819_${m_name}_all.log &
done

# Run it in the bias var tradeoff
for m_name in mlr ilr; do
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "mushroom" "credit" "heart" &> logs/082819_${m_name}_bias_var_all.log &
done

# Run the things in
python -u main.py --identifier 082319_datasets --overwrite 1 --model_name  lr mlr ilr --d_name heart &> logs/082819_lr_mlr_ilr_heart.log &


# Run the bias var tradeoff on heart & credit
for m_name in ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline; do
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --d_name heart credit &> logs/082819_heart_credit_all_${m_name}.log &
done

# Run everything for the heart
python -u main.py --identifier 082319_datasets --model_name ebm skgbt xgb-d1 xgb-d1-o8 ebm-o8 skgbt-o8 ebm-bf ebm-bf-o8 spline --d_name heart &> logs/082819_regenerate_heart.log &

# Run mushroom bias var tradeoff
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 --d_name mushroom  &> logs/082919_mushroom_all.log &

# Run adult for all spline
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name spline --d_name adult mushroom  &> logs/082919_adult_mushroom_spline.log &


# Rerun everything for mlr QQQQQQQ
python -u main.py --identifier 082319_datasets --model_name mlr --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "credit" "heart"  &> logs/082919_regenerate_mlr.log &
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name mlr --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "credit" "heart"  &> logs/082919_regenerate_bias_var_mlr.log &

# Rerun all the outer bagging for the skgbt and xgb for 082319_datasets QQ
for m_name in skgbt-o8 xgb-d1-o8; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "heart" "credit" &> logs/082919_regenerate_${m_name}.log &
done

# run a ilr without overwriting
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ilr --overwrite 0 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "credit" "heart"  &> logs/082919_regenerate_bias_var_ilr.log &

# Run new model in bias/var
# for m_name in rf-1000 xgb-d3 ; do
for d_name in "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "heart" "credit"; do
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name rf-n1000 xgb-d3 ebm-o16-i16 --overwrite 0 --d_name ${d_name}  &> logs/082919_new_models_bias_var_${d_name}.log &
done

# Try to finish adult and credit for all the model :)
# credit
for m_name in skgbt-o16 spline xgb-d1-o16; do
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --d_name credit &> logs/082819_credit_all_${m_name}.log &
done
# adult
for m_name in skgbt-o16 spline xgb-d1-o16; do
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --d_name adult &> logs/082819_adult_all_${m_name}.log &
done
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ebm ebm-bf ebm-bf-o16 ebm-o16 skgbt xgb-d1 --d_name adult &> logs/082819_adult_ebm_ebm-bf_ebm-bf-o16_ebm-o16_skgbt_xgb-d1.log &

python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name spline --d_name adult mushroom  &> logs/082919_adult_mushroom_spline.log &

# Rerun spline Orz.....
for m_name in spline; do
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --d_name adult credit &> logs/083019_annoying_${m_name}.log &
done

# Rerun a new model in bias_var: xgb-d3-o16
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name xgb-d3-o16 --overwrite 0 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "heart" "credit"  &> logs/083019_new_models_bias_var_all_d_xgb-d3-o16.log &


for m_name in xgb-d3 xgb-d3-o16 xgb-d1-o16 ebm-o16-i16 ebm-bf-o16-i16 rf-n1000; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "heart" "adult" "credit" &> logs/082919_regenerate_${m_name}.log &
done

# Rerun ilr for both bias/var and the datasets. Fixes the onehot encoder bugs....
python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ilr --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "heart" "credit"  &> logs/083019_new_models_bias_var_all_d_ilr.log &
python -u main.py --identifier 082319_datasets --model_name ilr --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "bikeshare" "churn" "adult" "heart" "credit"  &> logs/083019_all_d_ilr.log &

# run everything with the new dataset diabetes
for m_name in 'ebm' 'ebm-bf' 'ebm-bf-o16-i16' 'ebm-bf-o8' 'ebm-o16-i16' 'ilr' 'lr' 'mlr' ; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --overwrite 1 --d_name "diabetes" &> logs/083119_diabetes_${m_name}.log &
done

# 2nd batch
for m_name in 'rf-n1000' 'skgbt' 'skgbt-o8' 'spline' 'xgb-d1' 'xgb-d1-o16' 'xgb-d3' 'xgb-d3-o16'; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --overwrite 1 --d_name "diabetes" &> logs/083119_diabetes_${m_name}.log &
done

# run everything with calhousing dataset
# run everything with the new dataset diabetes
for m_name in 'ebm' 'ebm-bf' 'ebm-bf-o16-i16' 'ebm-bf-o8' 'ebm-o16-i16' 'ilr' 'lr' 'mlr' ; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "calhousing" "diabetes" &> logs/090119_calhousing_${m_name}.log &
done

# 2nd batch
for m_name in 'rf-n1000' 'skgbt' 'skgbt-o8' 'spline' 'xgb-d1' 'xgb-d1-o16' 'xgb-d3' 'xgb-d3-o16'; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "calhousing" "diabetes" &> logs/083119_calhousing_ "diabetes"${m_name}.log &
done



for m_name in 'ilr' 'lr' 'mlr'; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "churn" "adult" "heart" "credit" "diabetes" "bikeshare" "calhousing" &> logs/090119_${m_name}.log &
done

for m_name in 'ilr' 'lr' 'mlr'; do
    python -u main.py --identifier 082819_bias_var --exp_mode bias_var --model_name ${m_name} --overwrite 1 --d_name "pneumonia" "mimiciii" "breast" "churn" "adult" "heart" "credit" "bikeshare" &> logs/090119_vias_var_${m_name}.log &
done


## Now rerun things in the new datasets diabetes / calhousing
# for m_name in 'ebm' 'ebm-bf' 'ebm-bf-o16-i16' 'ebm-bf-o8' 'ebm-o16-i16' 'ilr' 'lr' 'mlr' ; do
for m_name in 'rf-n1000' 'skgbt' 'skgbt-o8' 'spline' 'xgb-d1' 'xgb-d1-o16' 'xgb-d3' 'xgb-d3-o16'; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "calhousing" "diabetes" &> logs/090119_rerun_${m_name}.log &
done


for d_name in "pneumonia" "mimiciii" "breast" "churn" "adult" "heart" "credit" "diabetes" "bikeshare" "calhousing"; do
    python feature_importances.py --identifier 0905 --d_name ${d_name} &> logs/090219_fimp_${d_name}.log &
done


for d_name in "mimiciii" "breast" "credit"; do
    python feature_importances.py --identifier 0905 --d_name ${d_name} &> logs/090219_fimp_${d_name}.log &
done


for d_name in "pneumonia" "mimiciii" "breast" "churn" "adult" "heart" "credit" "diabetes" "bikeshare" "calhousing"; do
    python feature_importances.py --identifier 0905 --d_name ${d_name} --model_name ebm-bf  &> logs/090219_fimp_ebmbf_${d_name}.log &
done

for d_name in "pneumonia" "mimiciii" "breast" "churn" "adult" "heart" "credit" "diabetes" "bikeshare" "calhousing"; do
    python feature_importances.py --identifier 0905_one_at_a_time --exp_mode OneAddATimeExp --d_name ${d_name}  &> logs/090319_fimp_${d_name}.log &
done


# Figure out diabetes. Rerun all the diabetes to fix the model
python -u main.py --identifier 082319_datasets --model_name ebm ebm-bf ebm-bf-o8 ebm-o16-i16 --overwrite 0 --d_name "diabetes" &> logs/090219_fixbugs_diabetes.log &

# 082319 has another problem :)
for m_name in ebm-o8 xgb-d1-o8; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "calhousing" "diabetes" &> logs/083119_calhousing_diabetes${m_name}.log &
done

# testing
m_name=ebm-o8
python -W error -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "calhousing" "diabetes"

# run the script to cache all the df (except the diabetes.... not finish yet.)
python -u summarize.py --identifier 0906 --d_name pneumonia mimiciii breast churn adult heart credit bikeshare calhousing --data_path results/082319_datasets.csv &> logs/0907_summarize.log &


# SS
for dset_name in "pneumonia" "mimiciii"; do
    for b in "0" "1"; do
        d_name="ss_${dset_name}_b${b}_r1377_spline"
        echo ${d_name};
        python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090619_semi_synthetic_${d_name}.log &
    done
done

# Analyze SS
python -u summarize.py --identifier 0906 --data_path results/090619_semi_synthetic.csv

# Rerun to cache the df....but credit keeps crashing Orz
python -u summarize.py --identifier 0906 --d_name bikeshare calhousing credit --data_path results/082319_datasets.csv &> logs/0907_summarize.log &

# Rerun SS on the old models to produce new columns for 0906_ss
for dset_name in "pneumonia" "mimiciii"; do
    for b in "0" "1"; do
        for produced_m_name in xgb-d1-o16 ebm-bf-o16-i16 ebm-o16-i16; do
            d_name="ss_${dset_name}_b${b}_r1377_${produced_m_name}"
            echo ${d_name};
            python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090819_semi_synthetic.log
        done
    done
done

# After SS finishes, run the caching for the SS df
python -u summarize.py --identifier 0907 --data_path results/090619_semi_synthetic.csv &> logs/090819_ss_summarize.log &
# And finish the feature importances plot
python feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0908_one_at_a_time --exp_mode OneAddATimeExp &> logs/090619_fimp_oneatatime_ss.log &
python feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0908_removal --exp_mode RemovalExp &> logs/090619_fimp_removal_ss.log &


## Rerun all the models in the new machine and ready to copy back
for m_name in xgb-d3-o16 spline xgb-d1-o8 ebm-bf-o16-i16; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "diabetes" &> logs/090819_rerun_${m_name}.log &
done
for m_name in spline ebm-bf-o16-i16; do
    python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "diabetes" &> logs/090819_rerun_${m_name}.log &
done

# rerun ss models for skgbt-o16
d_name=ss_mimiciii_b0_r1377_ebm-bf-o16-i16
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name skgbt-o16 --d_name ${d_name} &> logs/090919_semi_synthetic_${d_name}.log &

# Running.....wait.
d_name=ss_mimiciii_b0_r1377_ebm-bf-o16-i16
m_name=skgbt-o16
python feature_importances.py --data_path results/090619_semi_synthetic.csv --d_name ${d_name} --model_name ${m_name} --overwrite 1 --identifier 0908_one_at_a_time --exp_mode OneAddATimeExp &> logs/090819_fimp_oneatatime_ss.log &
python feature_importances.py --data_path results/090619_semi_synthetic.csv --d_name ${d_name} --model_name ${m_name} --overwrite 1 --identifier 0908_removal --exp_mode RemovalExp &> logs/090819_fimp_removal_ss.log &
python -u summarize.py --identifier 0907 --data_path results/090619_semi_synthetic.csv --d_name ${d_name} --model_name ${m_name} --overwrite 1 &> logs/090819_ss_summarize.log &

m_name=spline
python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "diabetes" &> logs/090819_rerun_${m_name}.log &

# Rerun these 4 datasets with all the models!!
for d_name in ss_pneumonia_b0_r1377_xgb-d1-o16 ss_mimiciii_b0_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16; do
    python -u main.py --identifier 090619_semi_synthetic --overwrite 1 --exp_mode training --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_rerun_semi_synthetic_${d_name}.log &
done

# Rerun all the datasets with these 2 models
for m_name in skgbt-o16 xgb-d1-o16; do
    python -u main.py --identifier 090619_semi_synthetic --overwrite 1 --exp_mode training --model_name ${m_name} --d_name \
        ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
        ss_mimiciii_b0_r1377_ebm-o16-i16 \
        ss_mimiciii_b0_r1377_spline \
        ss_mimiciii_b0_r1377_xgb-d1-o16 \
        ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
        ss_pneumonia_b0_r1377_ebm-o16-i16 \
        ss_pneumonia_b0_r1377_spline \
        ss_pneumonia_b0_r1377_xgb-d1-o16 \
        ss_pneumonia_b0_r1377_xgb-d1 \
        ss_mimiciii_b1_r1377_ebm-bf-o16-i16 \
        ss_mimiciii_b1_r1377_ebm-o16-i16 \
        ss_mimiciii_b1_r1377_spline \
        ss_mimiciii_b1_r1377_xgb-d1-o16 \
        ss_pneumonia_b1_r1377_ebm-bf-o16-i16 \
        ss_pneumonia_b1_r1377_ebm-o16-i16 \
        ss_pneumonia_b1_r1377_spline \
        ss_pneumonia_b1_r1377_xgb-d1-o16 \
        ss_pneumonia_b1_r1377_xgb-d1 \
     &> logs/090919_rerun_semi_synthetic_${m_name}.log &
done

# Wait to run these following 2 datasets
for d_name in ss_mimiciii_b0_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16; do
    python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_finish_semi_synthetic_${d_name}.log &
done

## Wait the result: run feature importances for the 082319_datasets
python -u feature_importances.py --data_path results/082319_datasets.csv --identifier 0909 --exp_mode RemovalExp --d_name mimiciii pneumonia adult breast churn heart credit &> logs/090919_fimp_removal.log &
python -u feature_importances.py --data_path results/082319_datasets.csv --identifier 0909 --exp_mode AddExp --d_name mimiciii pneumonia adult breast churn heart credit &> logs/090919_fimp_add.log &

# Run the semi synthetic with logistic regression
for dset_name in "pneumonia" "mimiciii"; do
    for b in "0"; do
        for produced_m_name in lr; do
            d_name="ss_${dset_name}_b${b}_r1377_${produced_m_name}"
            echo ${d_name};
            python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_semi_synthetic_${d_name}.log &
        done
    done
done

for dset_name in "pneumonia" "mimiciii"; do
    for b in "1"; do
        for produced_m_name in lr; do
            d_name="ss_${dset_name}_b${b}_r1377_${produced_m_name}"
            echo ${d_name};
            python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_semi_synthetic_${d_name}.log &
        done
    done
done

for dset_name in "mimiciii"; do
    for b in "1"; do
        for produced_m_name in lr; do
            d_name="ss_${dset_name}_b${b}_r1377_${produced_m_name}"
            echo ${d_name};
            python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_semi_synthetic_${d_name}.log &
        done
    done
done

for dset_name in "mimiciii"; do
    for b in "0"; do
        for produced_m_name in lr; do
            d_name="ss_${dset_name}_b${b}_r1377_${produced_m_name}"
            echo ${d_name};
            python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline --d_name ${d_name} &> logs/090919_semi_synthetic_${d_name}.log &
        done
    done
done

## Not finish the ss with "ss_pneumonia_b0_r1377_xgb-d1" "ss_pneumonia_b1_r1377_xgb-d1"
for m_name in ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 spline; do
    python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ${m_name} --d_name "ss_pneumonia_b0_r1377_xgb-d1" "ss_pneumonia_b1_r1377_xgb-d1" &> logs/091019_finish_semi_synthetic_${m_name}.log &
done

# Copy back from int server
scp int:/home/intelligible/kingsley/ebm-internship/models/090619_semi_synthetic/* ./models/090619_semi_synthetic/
scp int:/home/intelligible/kingsley/ebm-internship/datasets/ss_*_lr*.pkl ./datasets/
scp ./datasets/ss_*.pkl int:/home/intelligible/kingsley/ebm-internship/datasets/

# Speed up int for the lr on mimiciii with b0
d_name=ss_mimiciii_b0_r1377_lr
for m_name in ebm skgbt xgb-d1 xgb-d1-o16 ebm-o16 skgbt-o16 ebm-bf ebm-bf-o16 spline; do
    python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ${m_name} --d_name ${d_name} &> logs/090919_semi_synthetic_${m_name}_${d_name}.log &
done



# run full complexity model and mlr/ilr for ss. Just to see performance cap....
for m_name in rf-n1000 xgb-d3 xgb-d3-o16; do
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ${m_name} --d_name \
ss_pneumonia_b1_r1377_spline \
ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_xgb-d1-o16 \
ss_pneumonia_b0_r1377_ebm-o16-i16 \
ss_pneumonia_b0_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_spline \
ss_pneumonia_b0_r1377_spline \
ss_mimiciii_b1_r1377_spline \
ss_mimiciii_b1_r1377_xgb-d1-o16 \
ss_pneumonia_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_ebm-o16-i16 \
ss_pneumonia_b1_r1377_lr \
ss_mimiciii_b0_r1377_lr \
ss_pneumonia_b0_r1377_lr \
ss_mimiciii_b1_r1377_lr \
&> logs/091019_semi_synthetic_${m_name}_all.log &
done

for m_name in ilr mlr; do
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ${m_name} --d_name \
ss_pneumonia_b1_r1377_spline \
ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_xgb-d1-o16 \
ss_pneumonia_b0_r1377_ebm-o16-i16 \
ss_pneumonia_b0_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_spline \
ss_pneumonia_b0_r1377_spline \
ss_mimiciii_b1_r1377_spline \
ss_mimiciii_b1_r1377_xgb-d1-o16 \
ss_pneumonia_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_ebm-o16-i16 \
ss_pneumonia_b1_r1377_lr \
ss_mimiciii_b0_r1377_lr \
ss_pneumonia_b0_r1377_lr \
ss_mimiciii_b1_r1377_lr \
&> logs/091019_semi_synthetic_${m_name}_all.log &
done

## Now working on copying the 082319_datasets results back from int to the curretn server GCR
scp int:/home/intelligible/kingsley/ebm-internship/models/082319_datasets/* ./models/082319_datasets/

# Rerun summarize on these 2. Never able to do that due to memory constraints of credit+xgb-d1-o16
python -u summarize.py --identifier 0907 --d_name credit diabetes --data_path results/082319_datasets.csv &> logs/090919_summarize_credit_diabetes.log &

# Run the lr on other semi-synthetic datasets
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name lr --d_name \
ss_pneumonia_b1_r1377_spline \
ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_xgb-d1-o16 \
ss_pneumonia_b0_r1377_ebm-o16-i16 \
ss_pneumonia_b0_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_spline \
ss_pneumonia_b0_r1377_spline \
ss_mimiciii_b1_r1377_spline \
ss_mimiciii_b1_r1377_xgb-d1-o16 \
ss_pneumonia_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_ebm-o16-i16 \
&> logs/091019_semi_synthetic_lr_all.log &

python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ilr mlr --d_name \
ss_pneumonia_b1_r1377_spline \
ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_xgb-d1-o16 \
ss_pneumonia_b0_r1377_ebm-o16-i16 \
ss_pneumonia_b0_r1377_xgb-d1-o16 \
ss_mimiciii_b0_r1377_spline \
ss_pneumonia_b0_r1377_spline \
ss_mimiciii_b1_r1377_spline \
ss_mimiciii_b1_r1377_xgb-d1-o16 \
ss_pneumonia_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-o16-i16 \
ss_mimiciii_b1_r1377_ebm-bf-o16-i16 \
ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
ss_pneumonia_b1_r1377_ebm-o16-i16 \
&> logs/091019_semi_synthetic_ilr_mlr_all.log &

python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name lr --d_name \
ss_pneumonia_b1_r1377_lr \
ss_mimiciii_b0_r1377_lr \
ss_pneumonia_b0_r1377_lr \
ss_mimiciii_b1_r1377_lr \
&> logs/091019_semi_synthetic_lr_all2.log &

python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ilr mlr --d_name \
ss_pneumonia_b1_r1377_lr \
ss_mimiciii_b0_r1377_lr \
ss_pneumonia_b0_r1377_lr \
ss_mimiciii_b1_r1377_lr \
&> logs/091019_semi_synthetic_ilr_mlr_all2.log &


# Orz....Rerun the spline for ebm-bf-o16-i16
python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b0_r1377_ebm-bf-o16-i16 \
&> logs/091119_fix_semi_synthetic_spline.log &

python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b1_r1377_ebm-o16-i16 \
&> logs/091119_fix_semi_synthetic_spline2.log &

python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b0_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline3.log &

python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b1_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline4.log &

python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_pneumonia_b0_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline5.log &

python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_pneumonia_b1_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline6.log &

for d_name in ss_pneumonia_b0_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b0_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16; do
python -u main.py --overwrite 1 --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
${d_name} \
&> logs/091119_fix_semi_synthetic_spline_${d_name}.log &
done

## Move to int server to run. So SLOWWWWWW Orz.....
python -u main.py --overwrite 1 --identifier 091119_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b1_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline4.log &

d_name=ss_mimiciii_b1_r1377_xgb-d1-o16
python -u main.py --overwrite 1 --identifier 091119_semi_synthetic --exp_mode training --model_name spline --d_name \
${d_name} \
&> logs/091119_fix_semi_synthetic_spline_${d_name}.log &

python -u quick_fix_spline.py --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b1_r1377_lr \
&> logs/091119_fix_semi_synthetic_spline4.log &

d_name=ss_mimiciii_b1_r1377_xgb-d1-o16
python -u quick_fix_spline.py --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
${d_name} \
&> logs/091119_fix_semi_synthetic_spline_${d_name}.log &

python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b1_r1377_lr

d_name=ss_mimiciii_b1_r1377_xgb-d1-o16
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name spline --d_name \
${d_name}




# Orz.....Fix the bug on int server
m_name=xgb-d3-o16
python -u main.py --identifier 090619_semi_synthetic --exp_mode training --model_name ${m_name} --d_name \
ss_mimiciii_b1_r1377_xgb-d1-o16 \
&> logs/091119_debug_bagging.log &


## Spline performs wierd on ss_mimiciii_b0_r1377_ebm-o16-i16, rerun
python -u main.py --overwrite 1 --identifier 091119_semi_synthetic --exp_mode training --model_name spline --d_name \
ss_mimiciii_b0_r1377_ebm-o16-i16 \
&> logs/091119_debug_spline_ss_mimic.log &




# for d_name in \
#     ss_pneumonia_b1_r1377_spline ss_pneumonia_b0_r1377_spline ss_mimiciii_b1_r1377_spline ss_mimiciii_b0_r1377_spline ss_pneumonia_b0_r1377_ebm-bf-o16-i16 \
#     ss_pneumonia_b0_r1377_ebm-o16-i16 ss_pneumonia_b1_r1377_ebm-bf-o16-i16 ss_pneumonia_b1_r1377_ebm-o16-i16 ss_mimiciii_b0_r1377_ebm-bf-o16-i16 ss_mimiciii_b0_r1377_ebm-o16-i16 \
#     ss_mimiciii_b1_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16 ss_pneumonia_b0_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b0_r1377_xgb-d1-o16 \
#     ss_mimiciii_b1_r1377_xgb-d1-o16 ss_pneumonia_b0_r1377_lr ss_mimiciii_b0_r1377_lr ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr; do

#     python -u summarize.py --identifier 0910 \
#     --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 spline xgb-d1-o16 skgbt-o16 lr \
#     --data_path results/090619_semi_synthetic.csv &> logs/091019_resummarize_ss_${d_name}.log &
# done

# Skip spline for now......
python -u summarize.py --identifier 0910 \
    --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 xgb-d1-o16 skgbt-o16 lr ilr mlr \
    --data_path results/090619_semi_synthetic.csv &> logs/091019_resummarize_ss.log &

python -u feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0910 --exp_mode RemovalExp --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 xgb-d1-o16 skgbt-o16 lr ilr mlr spline &> logs/091119_fimp_removal_ss.log &
python -u feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0910 --exp_mode AddExp --model_name ebm skgbt xgb-d1 ebm-o16 ebm-bf ebm-bf-o16 xgb-d1-o16 skgbt-o16 lr ilr mlr spline &> logs/091119_fimp_add_ss.log &

## Later
python -u summarize.py --overwrite 1 --identifier 0910 \
    --model_name spline \
    --data_path results/090619_semi_synthetic.csv &> logs/091019_resummarize_ss2.log &

python -u feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0910 --exp_mode RemovalExp --model_name spline &> logs/091119_fimp_removal_ss2.log &
python -u feature_importances.py --data_path results/090619_semi_synthetic.csv --identifier 0910 --exp_mode AddExp --model_name spline &> logs/091119_fimp_add_ss2.log &






################# Maybe Investigate ##################
## 1. Spline on diabetes never converge: SVD did not converge.
## TOWAIT: Rerun the diabetes with spline in int server
m_name=spline
python -u main.py --identifier 082319_datasets --model_name ${m_name} --d_name "diabetes" &> logs/090819_rerun_${m_name}.log &



scp int:/home/intelligible/kingsley/ebm-internship/models/090619_semi_synthetic/ss_*ilr*.pkl ./models/090619_semi_synthetic/
scp int:/home/intelligible/kingsley/ebm-internship/models/090619_semi_synthetic/ss_*mlr*.pkl ./models/090619_semi_synthetic/


# Rerun heart


# Run on the int server
python -u main.py --overwrite 1 --identifier 082319_datasets --model_name ebm skgbt xgb-d1 ebm-o8 ebm-bf ebm-bf-o8 spline skgbt-o8 xgb-d1-o8 xgb-d3 rf-n1000 ebm-o16-i16 xgb-d3-o16 xgb-d1-o16 ebm-bf-o16-i16 lr mlr ilr --d_name heart &> logs/091219_rerun_heart.log &

## WAIT after the heart finishes
python -u summarize.py --overwrite 1 --identifier 0907 --d_name heart --data_path results/082319_datasets.csv &> logs/091219_summarize_heart.log &
python -u feature_importances.py --overwrite 1 --data_path results/082319_datasets.csv --identifier 0909 --exp_mode RemovalExp --d_name heart &> logs/091219_fimp_heart_removal.log &
python -u feature_importances.py --overwrite 1 --data_path results/082319_datasets.csv --identifier 0909 --exp_mode AddExp --d_name heart &> logs/091219_fimp_heart_add.log &

scp int:/home/intelligible/kingsley/ebm-internship/models/082319_datasets/heart*.pkl ./models/082319_datasets/


## 0913
# Run an auc metrics for increasing features in adding experiments....
for d_name in 'adult' 'bikeshare' 'breast' 'calhousing' 'churn' 'credit' 'mimiciii' 'pneumonia' 'heart'; do
python -u feature_importances.py --metric auc --data_path results/082319_datasets.csv --identifier 0913-auc --exp_mode AddExp --d_name ${d_name} --model_name ebm ebm-bf lr xgb-d1 skgbt spline mlr ilr &> logs/091319_fimp_add_${d_name}.log &
done

d_name=diabetes
python -u feature_importances.py --metric auc --data_path results/082319_datasets.csv --identifier 0913-auc --exp_mode AddExp --d_name ${d_name} --n_features_limit 100 --model_name ebm ebm-bf lr xgb-d1 skgbt spline mlr ilr &> logs/091319_fimp_add_${d_name}.log &


# Analyze the table's sensitivity for normalized score: change the baseline as 0.5 for auc....

# Rerun the summarization for ss




################# TODO ###################
## The XGB purification.....
## Figure out why House Age is good in the cal housing dataset
## Spline performs wierd on ss_mimiciii_b0_r1377_ebm-o16-i16, even rerun can replicate. Why???????????????????????????

## EBM-o16-i16 ss datasets
counter=1
for m_name in "xgb-d3-o16 ebm skgbt xgb-d1 ebm-o8" "xgb-d3-o16 ebm-bf ebm-bf-o8 mlr ilr" "spline skgbt-o8 xgb-d1-o8 xgb-d3 rf-n1000 ebm-o16-i16" "xgb-d1-o16 ebm-bf-o16-i16"; do
python -u main.py --identifier 090619_semi_synthetic --model_name ${m_name} --d_name ss_pneumonia_b0_r1377_ebm-o16-i16-it10 ss_mimiciii_b0_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 &> logs/091219_run_ss_ebm-it10-${counter}.log &
echo $((counter++))
done

for m_name in "xgb-d3-o16 ebm skgbt xgb-d1 ebm-o8" "xgb-d3-o16 ebm-bf ebm-bf-o8 mlr ilr" "spline skgbt-o8 xgb-d1-o8 xgb-d3 rf-n1000 ebm-o16-i16" "xgb-d1-o16 ebm-bf-o16-i16"; do
    echo ${m_name}
done

# Generate the dataset first!!
for d_name in ss_pneumonia_b0_r1377_ebm-o16-i16-it10 ss_mimiciii_b0_r1377_ebm-o16-i16-it10; do
python -u main.py --identifier 090619_semi_synthetic --model_name lr \
--d_name ${d_name} \
&> logs/091219_gen_dataset_${d_name}.log &
done
()

# Run a EBM-bf-o16 to compare?



# Retrain a one whole model to visualize the graph? :)
# But if so, we don't have a error bar......

# Resummarize due to the extrapolation error
python -u summarize.py --overwrite 1 --identifier 0910 \
    --data_path results/090619_semi_synthetic.csv &> logs/091519_resummarize_ss.log &

python -u summarize.py --overwrite 1 --identifier 0907 \
    --data_path results/082319_datasets.csv &> logs/091519_resummarize_everything.log &

# Resummarize ss to get the ground truth metrics with the weights....
python -u summarize.py --overwrite 1 --identifier 0910 \
    --model_name gnd_truth --data_path results/090619_semi_synthetic.csv &> logs/091619_resummarize_ss.log &


bikeshare breast pneumonia mimiciii churn adult
# Credit dies. Rerun resummarization on others
python -u summarize.py --overwrite 1 --identifier 0907 \
    --d_name calhousing heart credit  \
    --data_path results/082319_datasets.csv &> logs/091519_resummarize_everything.log &

# Rerun ss and datasets with only 15% test set size....
counter=1
for m_name in "xgb-d3-o20 xgb-d1-o20 skgbt-o20 ebm-o30" "ebm-o30-i30 ebm-bf-o30-i30 ebm-bf-o30 skgbt-o20 rf-n1000 mlr lr ilr spline"; do
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 --model_name ${m_name} --d_name ss_pneumonia_b0_r1377_ebm-o16-i16-it10 ss_mimiciii_b0_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 &> logs/091219_run_ss_ebm-it10-${counter}.log &
echo $((counter++))
done

# Rerun the ss since it dies with the spline prediction of nan QQ.....
counter=2
m_name="ebm-o30-i30 ebm-bf-o30-i30 ebm-bf-o30 skgbt-o20 rf-n1000 mlr lr ilr spline"
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 --model_name ${m_name} --d_name ss_pneumonia_b1_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-bf-o16-i16 ss_pneumonia_b1_r1377_spline ss_mimiciii_b1_r1377_spline ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16 &> logs/091719_run_ss_all-${counter}.log &

# Rerun the heart metric
python -u main.py --identifier 082319_datasets --model_name ebm skgbt xgb-d1 ebm-o8 ebm-bf ebm-bf-o8 spline skgbt-o8 xgb-d1-o8 xgb-d3 rf-n1000 ebm-o16-i16 xgb-d3-o16 xgb-d1-o16 ebm-bf-o16-i16 lr mlr ilr --d_name heart &> logs/092519_rerun_heart.log &

# Rerun
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 --model_name skgbt-o20 --d_name ss_pneumonia_b1_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-bf-o16-i16 ss_pneumonia_b1_r1377_spline ss_mimiciii_b1_r1377_spline ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16 &> logs/092719_rerun_skgbt-o20.log &


# TODO: rerun xgb-d3 and skgbt-d3
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 --d_name ss_pneumonia_b1_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-bf-o16-i16 ss_pneumonia_b1_r1377_spline ss_mimiciii_b1_r1377_spline ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16 &> logs/092719_run_ss_more.log &
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 \
--d_name breast bikeshare churn adult pneumonia calhousing heart mimiciii credit &> logs/092719_run_more.log &


# TODO: Check why ebm-bf-o30 performs worse in credit dataset??????


# Rerun summarization...
python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv &> logs/092619_summarize.log &
python -u summarize.py --identifier 0927 \
    --data_path results/091719_semi_synthetic.csv &> logs/092619_summarize_ss.log &

# Rerun the removal experiments and adding experiments on new models and
python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --identifier 0927 --exp_mode AddExp &> logs/092519_fimp_add_all_datasets.log &
python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --identifier 0927 --exp_mode RemovalExp &> logs/092519_fimp_removal_all_datasets.log &

# GIVEUP: since it's hard to estimate the PDP!!! Rerun the removal experiments and summarize for new models
python -u summarize.py --overwrite 1 --identifier 0927 --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 \
    --data_path results/091719_datasets.csv &> logs/092819_summarize_new_models.log &
python -u summarize.py --overwrite 1 --identifier 0927 --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 \
    --data_path results/091719_semi_synthetic.csv &> logs/092819_summarize_ss_new_models.log &

python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 --identifier 0927 --exp_mode AddExp &> logs/092819_fimp_add_all_datasets_new_models.log &
python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --model_name xgb-d3 skgbt-d3 skgbt-d3-o20 --identifier 0927 --exp_mode RemovalExp &> logs/092819_fimp_removal_all_datasets_new_models.log &

# TODO: featuer_importances with full complexity models???

# Work on: fused lasso. Skip the 3 datasets....
for d_name in breast bikeshare churn pneumonia calhousing heart mimiciii adult; do
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name flam --overwrite 1 \
--d_name ${d_name}  &> logs/1011_flam_${d_name}.log &
done

for d_name in ss_pneumonia_b1_r1377_ebm-bf-o16-i16 ss_mimiciii_b1_r1377_ebm-bf-o16-i16 ss_pneumonia_b1_r1377_spline ss_mimiciii_b1_r1377_spline ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr ss_pneumonia_b1_r1377_xgb-d1-o16 ss_mimiciii_b1_r1377_xgb-d1-o16 ss_pneumonia_b1_r1377_ebm-o16-i16-it10 ss_mimiciii_b1_r1377_ebm-o16-i16-it10 ss_pneumonia_b1_r1377_ebm-o16-i16 ss_mimiciii_b1_r1377_ebm-o16-i16; do
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
--model_name flam --d_name ${d_name} &> logs/101119_run_ss_flam_${d_name}.log &
done

# Summarize these
python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv &> logs/100919_summarize.log &
python -u summarize.py --identifier 0927 \
    --data_path results/091719_semi_synthetic.csv &> logs/100919_summarize_ss.log &

python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --model_name flam --identifier 0927 --exp_mode AddExp &> logs/100919_fimp_add_all_datasets_new_models.log &
python -u feature_importances.py --metric mse --data_path results/091719_datasets.csv --model_name flam --identifier 0927 --exp_mode RemovalExp &> logs/100919_fimp_removal_all_datasets_new_models.log &


for d_name in credit diabetes; do
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name flam --overwrite 1 \
--d_name ${d_name}  &> logs/1011_flam_${d_name}.log &
done

# Regenerate all the skgbt-o20
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name skgbt-o20 \
--d_name breast bikeshare churn pneumonia calhousing heart mimiciii adult credit diabetes

# Regenerate the SS
python -u summarize.py --identifier 0927 \
--data_path results/091719_datasets.csv --model_name skgbt-o20 flam --overwrite 1


# Forget to run diabetes....
counter=1
for model_name in 'xgb-d3-o20 ebm-o30-i30 ebm-bf-o30-i30 ebm-bf-o30' 'mlr lr ilr ebm-o30 xgb-d3 xgb-d1-o20 rf-n1000' \
'skgbt-d3 skgbt-d3-o20 skgbt-o20'; do
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name ${model_name} \
--d_name diabetes &> logs/1011_forget_run_diabetes_${counter}.log &
echo $((counter++))
done

counter=2
model_name='mlr lr ilr ebm-o30 xgb-d3 xgb-d1-o20 rf-n1000'
python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name ${model_name} \
--d_name diabetes &> logs/1011_forget_run_diabetes_${counter}.log &

# Run a ss by flam
for d_name in ss_pneumonia_b1_r1377_flam ss_mimiciii_b1_r1377_flam; do
python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
--model_name ebm-o30-i30 ebm-bf-o30-i30 ebm-bf-o30 rf-n1000 mlr lr ilr spline xgb-d3-o20 xgb-d1-o20 ebm-o30 skgbt-o20 xgb-d3 skgbt-d3 skgbt-d3-o20 flam \
--d_name ${d_name} &> logs/101419_run_all_models_on_${d_name}.log &
done

# Run summarization for 2 new ss datasets flam
python -u summarize.py --identifier 0927 \
    --data_path results/091719_semi_synthetic.csv \
    --d_name ss_pneumonia_b1_r1377_flam ss_mimiciii_b1_r1377_flam &> logs/101619_summarize_ss.log &

# Run the increment datasets!
for inc_ratio in 1 3; do
    for d_name in ss_pneumonia_b1_r1377_spline_inc${inc_ratio} ss_pneumonia_b1_r1377_ebm-o16-i16_inc${inc_ratio}; do
        echo ${d_name}
        python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
        --model_name ebm-o30-i30 ebm-bf-o30 mlr lr ilr spline xgb-d1-o20 skgbt-o20 flam --d_name ${d_name} &> logs/101619_run_ss_inc_${d_name}.log &
    done
done

python -u summarize.py --identifier 0927 \
    --data_path results/091719_semi_synthetic.csv \
    --d_name ss_pneumonia_b1_r1377_ebm-o16-i16_inc1 ss_pneumonia_b1_r1377_ebm-o16-i16_inc3 ss_pneumonia_b1_r1377_spline_inc1 ss_pneumonia_b1_r1377_spline_inc3 \
    &> logs/101719_summarize_inc_ss.log &

# Run a 5
for inc_ratio in 5; do
    for d_name in ss_pneumonia_b1_r1377_spline_inc${inc_ratio} ss_pneumonia_b1_r1377_ebm-o16-i16_inc${inc_ratio}; do
        echo ${d_name}
        ( \
            (python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
                --model_name ebm-o30-i30 ebm-bf-o30 mlr lr ilr spline xgb-d1-o20 skgbt-o20 flam \
                --d_name ${d_name} &> logs/101719_run_ss_inc_${d_name}.log ) \
        && \
        (python -u summarize.py --identifier 0927 \
            --data_path results/091719_semi_synthetic.csv --d_name ${d_name} &> logs/101719_summarize_inc_${d_name}.log) \
        ) &
    done
done

for inc_ratio in 1 3 5; do
    for d_name in ss_pneumonia_b1_r1377_xgb-d1-o16_inc${inc_ratio}; do
        echo ${d_name}
        ( \
            (python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
                --model_name ebm-o30-i30 ebm-bf-o30 mlr lr ilr spline xgb-d1-o20 skgbt-o20 flam \
                --d_name ${d_name} &> logs/101719_run_ss_inc_${d_name}.log ) \
        && \
        (python -u summarize.py --identifier 0927 \
            --data_path results/091719_semi_synthetic.csv --d_name ${d_name} &> logs/101719_summarize_inc_${d_name}.log) \
        ) &
    done
done


for inc_ratio in 10; do
    for d_name in ss_pneumonia_b1_r1377_xgb-d1-o16_inc${inc_ratio} ss_pneumonia_b1_r1377_spline_inc${inc_ratio} ss_pneumonia_b1_r1377_ebm-o16-i16_inc${inc_ratio}; do
        echo ${d_name}
        ( \
            (python -u main.py --identifier 091719_semi_synthetic --test_size 0.15 \
                --model_name ebm-o30-i30 ebm-bf-o30 mlr lr ilr spline xgb-d1-o20 skgbt-o20 flam \
                --d_name ${d_name} &> logs/101719_run_ss_inc_${d_name}.log ) \
        && \
        (python -u summarize.py --identifier 0927 \
            --data_path results/091719_semi_synthetic.csv --d_name ${d_name} &> logs/101719_summarize_inc_${d_name}.log) \
        ) &
    done
done

# All the summary df are gone QAQ. Resummarize everything!
python -u summarize.py --identifier 0927 \
    --data_path results/091719_semi_synthetic.csv \
    &> logs/102119_resummarize_ss.log &

counter=1
for d_name in 'adult bikeshare breast calhousing mimiciii' 'churn pneumonia heart' 'credit' 'diabetes'; do
python -u summarize.py --identifier 0927 --data_path results/091719_datasets.csv \
--d_name ${d_name} &> logs/102119_resummarize_datasets_${counter}.log &
echo $((counter++))
done


# Rerun diabetes.....
counter=3
model_name='skgbt-d3 skgbt-d3-o20 skgbt-o20'
((python -u main.py --identifier 091719_datasets --test_size 0.15 --model_name ${model_name} \
--d_name diabetes &> logs/1022_forget_run_diabetes_${counter}.log) \
&& (python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv --model_name ${model_name} --d_name diabetes  &> logs/1022_forget_run_diabetes_summarize_${counter}.log)) &

# Run the cv for GBT and XGB.  No changes to ebm, ebm-bf, lr, flam.
# Spline does not seem to worth it, especially this package is so unstable and it gets best perf
for model_name in skgbt-d1-o20-cv xgb-d1-o20-cv; do
((python -u main.py --overwrite 1 --identifier 091719_datasets --test_size 0.15 --model_name ${model_name} \
--d_name pneumonia mimiciii &> logs/1102_cv_${model_name}.log) \
&& (python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv --model_name ${model_name} --d_name pneumonia mimiciii &> logs/1102_cv_summarize_${model_name}.log)) &
done


# Rerun everything with just 5 times random splits...
'xgb-d3-o20', 'ebm-o30-i30', 'ebm-bf-o30-i30', 'ebm-bf-o30',
       'xgb-d1-o20', 'rf-n1000', 'mlr-o10', 'lr-o10', 'ilr-o10', 'ebm-o30', 'spline-o10',
       'xgb-d3', 'xgb-d3-20', 'skgbt-d3', 'skgbt-d3-o20', 'flam-o10', 'skgbt-o20',
       'xgb-d1-o20-cv', 'skgbt-d1-o20-cv'

'spline'

counter=1
for model_name in 'xgb-d1-o20-cv skgbt-d1-o20-cv ebm-o50-i50 lr-o20 ebm-bf-o50' 'ebm-bf-o50-i50 mlr-o20 ilr-o20 flam-o20'; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name mimiciii pneumonia bikeshare breast calhousing churn heart &> logs/1104_larger_run_${counter}.log &
    echo $((counter++))
done

for model_name in spline-cv; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name mimiciii pneumonia bikeshare breast calhousing churn heart &> logs/1104_larger_run_${model_name}.log &
done

# Ready to run the larger datasets if working...
for model_name in 'xgb-d1-o20-cv skgbt-d1-o20-cv lr-o20' 'mlr-o20 ilr-o20 flam-o20'; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name mimiciii pneumonia bikeshare breast calhousing churn heart &> logs/1104_larger_run_${counter}.log &
    echo $((counter++))
done

model_name='mlr-o20 ilr-o20'
counter=2
python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
    --d_name mimiciii pneumonia bikeshare breast calhousing churn heart &> logs/1104_larger_run_${counter}.log &

credit diabetes adult

# Forget to run ebm-bf-o50-i50

counter=1
for model_name in "ebm-bf-o50 ebm-o50-i50" "skgbt-d1-o20-cv xgb-d1-o20-cv"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name mimiciii pneumonia bikeshare breast calhousing churn heart adult credit diabetes &> logs/1106_larger_run_${counter}.log &
    echo $((counter++))
done

## Run testing to check lam range for spline! (n_spline=35)
for d_name in adult mimiciii pneumonia bikeshare breast calhousing churn heart diabetes credit; do
    python -u testing.py --d_name ${d_name} &> logs/1107_lam_range_check_${d_name} &
done

## Run spline-v2 in all datasets
for model_name in "spline-v2"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name pneumonia bikeshare breast calhousing churn heart mimiciii credit adult diabetes &> logs/1108_larger_run_${model_name}.log &
done


## TORUN (after checking the spline lam range...): Run a ss (generated by full model)
# Model list:
'ebm-o50-i50', 'ebm-bf-o50', 'xgb-d1-o20-cv', 'mlr', 'lr', 'ilr', 'spline-v2', 'xgb-d3-o20-cv', 'flam'
# Diff Models: 'skgbt-d3-cv', 'skgbt-d3-o20-cv', 'skgbt-d1-o20-cv'


## WAITING!!!!
counter=1
for d_name in 'ss_pneumonia_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-o50-i50' 'ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_spline-v2 ss_mimiciii_b1_r1377_spline-v2'; do
    python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr spline-v2 xgb-d3-o20-cv flam \
        --d_name ${d_name} &> logs/1108_ss_${counter}.log &
    echo $((counter++))
done

# Rerun the last part
for model_name in "skgbt-d1-o20-cv" "xgb-d1-o20-cv"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name mimiciii pneumonia bikeshare breast calhousing churn heart adult credit diabetes &> logs/1111_larger_run_${model_name}.log &
done

# See the CV results
# 1. XGB and EBM w/ larger bagging consistenly improves!
# 2. (TODO) SKGBT still has wierd perfomrnace in the credit datasets.... maybe tuning a bit more? SKGBT performance are similar
# See the SS results
# 1. (TODO) weighted log probability

for d_name in 'ss_pneumonia_b1_r1377_flam' 'ss_mimiciii_b1_r1377_flam'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr spline-v2 xgb-d3-o20-cv flam \
        --d_name ${d_name} &> logs/1108_ss_${d_name}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/1112_ss_summarize_${d_name}.log) &
done

for d_name in 'ss_pneumonia_b1_r1377_flam ss_mimiciii_b1_r1377_flam'; do
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/1112_ss_summarize_3.log &
done

counter=1
for d_name in 'ss_pneumonia_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-o50-i50' 'ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_spline-v2 ss_mimiciii_b1_r1377_spline-v2'; do
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/1112_ss_summarize_${counter}.log &
    echo $((counter++))
done

# Rerun the skgbt-d1-o20-cv in credit?
for model_name in "skgbt-d1-o20-cv-v2"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name credit &> logs/1112_larger_run_${model_name}.log &
done


# Wait the spline-v2 and skgbt-v2 to finish. Then we can summarize altogether :)
for model_name in "ebm-bf-o50 ebm-o50-i50" "skgbt-d1-o20-cv xgb-d1-o20-cv" "spline-v2" "skgbt-d1-o20-cv" "xgb-d1-o20-cv"; do

# TODO:
# 1. Finish the weighted log probability for calibration
# 2. Summarize the new model :)
# 3. Check other experiments like simple simulations for jumps or A-B1-B2 experiments....
#    Maybe do another experiment to show under noise settings, the EBM will perform better.

# Unclear why L2 is usually better than L1 in linear regression, while EBM and EBM-BF are very similar...
# Maybe design an experiment with lots of noisy small features and test l1, l2 and EBM, EBM-BF?


# Test new datasets for their spline ranges!
for d_name in support2reg support2cls onlinenews wine; do
for d_name in onlinenewscls; do
for d_name in mimicii; do
    python -u testing.py --d_name ${d_name} &> logs/1107_lam_range_check_${d_name} &
done


for d_name in 'ss_pneumonia_b1_r1377_lr' 'ss_mimiciii_b1_r1377_lr'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr spline-v2 xgb-d3-o20-cv flam \
        --d_name ${d_name} &> logs/1108_ss_${d_name}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/1112_ss_summarize_${d_name}.log) &
done

# Run more datasets: support2reg support2cls onlinenews wine (mimicii?)
#

counter=1
for d_name in 'support2reg support2cls mimicii' 'onlinenews wine'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name xgb-d1-o20-cv xgb-d3-o20 xgb-d3-o20-cv rf-n1000 mlr lr ilr flam skgbt-d1-o20 skgbt-d1-o20-cv ebm-o50-i50 ebm-bf-o50 spline-v2 \
        --d_name ${d_name} &> logs/1113_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1113_summarize_${counter}.log) &
    echo $((counter++))
done

for model_name in xgb-d3-o20-cv xgb-d3; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name bikeshare calhousing support2reg onlinenews wine &> logs/1117_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name bikeshare calhousing support2reg onlinenews wine &> logs/1117_summarize_${model_name}.log) &
done


# TODO:
# 1. Check the new datasets performance
# 2. Noisy experiments settings...

# Run a ebm with hyperparameter search
counter=1
for model_name in 'ebm-o50-i50-cv ebm-bf-o50-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name heart pneumonia wine &> logs/1118_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name heart pneumonia wine &> logs/1118_summarize_${counter}.log) &
    echo $((counter++))
done

counter=1
for model_name in 'ebm-o50-i50-cv ebm-bf-o50-cv'; do
    (python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name heart pneumonia wine &> logs/1118_summarize_${counter}.log) &
    echo $((counter++))
done

counter=1
for model_name in 'ebm-o50-i50-cv ebm-bf-o50-cv'; do
    (python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name  &> logs/1120_ebms_${counter}.log) &
done

####### EBM!!!!! #########
counter=1
for model_name in 'ebm-o50-i50-cv ebm-bf-o50-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name breast churn mimicii mimiciii &> logs/1121_ebms_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name breast churn support2cls mimicii &> logs/1121_ebms_summarize_${counter}.log) &
    echo $((counter++))
done

# counter=1
for model_name in 'ebm-bf-o50-i50'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name breast churn heart mimicii pneumonia support2cls mimiciii wine &> logs/1121_ebms_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name breast churn support2cls mimicii &> logs/1121_ebms_summarize_${model_name}.log) &
    # echo $((counter++))
done

####### XGB
counter=1
for model_name in 'xgb-d1-o20 xgb-d3'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name mimicii support2cls &> logs/1121_xgbs_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name mimicii support2cls &> logs/1121_xgbs_summarize_${counter}.log) &
    echo $((counter++))
done

counter=1
for model_name in 'xgb-d3-o20-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name breast heart churn mimiciii pneumonia &> logs/1121_xgbs_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name breast heart churn mimiciii pneumonia &> logs/1121_xgbs_summarize_${model_name}.log) &
    # echo $((counter++))
done

######## SKGBT
d_name="adult breast heart churn pneumonia support2cls mimicii mimiciii"
counter=1
for model_name in 'skgbt-d3-cv' 'skgbt-d3-o20-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1121_skgbts_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1121_skgbts_summarize_${model_name}.log) &
    echo $((counter++))
done

# Run skgbt-d3-cv on credit, diabetes
d_name="credit diabetes"
for model_name in 'skgbt-d3-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1127_skgbts_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1127_skgbts_summarize_${model_name}.log) &
done

# Try to run spline in diabetes
for model_name in "spline-v2"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name diabetes &> logs/1108_larger_run_${model_name}.log &
done

for model_name in "flam"; do
    python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name diabetes &> logs/1108_larger_run_${model_name}.log &
done


## For regression, run more things
d_name="bikeshare calhousing onlinenews support2reg"
counter=1
for model_name in 'ebm-o50-i50-cv ebm-o50-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1128_ebms_reg_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1128_ebms_reg_summarize_${counter}.log) &
    echo $((counter++))
done

# skgbt
d_name="bikeshare calhousing onlinenews support2reg wine"
counter=1
for model_name in 'skgbt-d3-cv' 'skgbt-d3-o20-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1128_skgbts_reg_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1128_skgbts_reg_summarize_${counter}.log) &
    echo $((counter++))
done


# Have not run: Run bias/var tradeoff on these new models:
#
counter=1
for model_name in 'ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr rf-n1000 spline-v2 xgb-d1-o20-cv xgb-d3 xgb-d3-o20'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg onlinenews support2cls mimicii mimiciii wine adult diabetes credit \
        --model_name ${model_name} &> logs/112819_bias_var_${counter}.log &
    echo $((counter++))
done


# Rerun in int2....
# All the skgbts
d_name="bikeshare calhousing onlinenews support2reg wine"
counter=1
for model_name in 'skgbt-d3-cv' 'skgbt-d3-o20-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1128_skgbts_reg_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1128_skgbts_reg_summarize_${counter}.log) &
    echo $((counter++))
done

d_name="adult breast heart churn pneumonia support2cls mimicii mimiciii"
counter=1
for model_name in 'skgbt-d3-cv' 'skgbt-d3-o20-cv'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1121_skgbts_datasets_${model_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1121_skgbts_summarize_${model_name}.log) &
    echo $((counter++))
done


# xgb-d3-o20 in onlinewnews pop up nan? Dies in xgb-d3-o20
###### rerun without onlinenews dataset
# Rerun bias/var counter1 in int, counter2 in int2
counter=1
for model_name in 'ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr rf-n1000 spline-v2' ; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult diabetes credit \
        --model_name ${model_name} &> logs/120419_bias_var_${counter}.log &
    echo $((counter++))
done

## int
model_name='ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr'
counter=1
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/120419_bias_var_${counter}.log &

model_name='spline-v2'
counter=3
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/120419_bias_var_${counter}.log &

## int2
model_name='xgb-d1-o20-cv xgb-d3 xgb-d3-o20 rf-n1000'
counter=2
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/120419_bias_var_${counter}.log &


## int2 => Finish the onlinenews dataset
model_name='xgb-d1-o20-cv xgb-d3 rf-n1000 ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr rf-n1000 spline-v2'
counter=1
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name onlinenews \
        --model_name ${model_name} &> logs/120819_bias_var_onlinenews_${counter}.log &

## int2 => Running: have not run skgbt-d3-o20-cv on credit/diabetes
model_name='skgbt-d3-o20-cv'
counter=1
for d_name in 'credit' 'diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/1208_skgbts_datasets_${model_name}_${d_name}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/1208_skgbts_summarize_${model_name}_${d_name}.log) &
    echo $((counter++))
done


## int: Run the removal and adding experiment for these new models???
## Figure out how to take the removal and add experiments: use only the gam model
model_name='ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr skgbt-d1-o20-cv spline-v2 xgb-d1-o20-cv'
python feature_importances.py --data_path results/091719_datasets.csv --model_name ${model_name} --end_splits 5 --identifier 1208_removal --exp_mode RemovalExp &> logs/120819_fimp_removal.log &
python feature_importances.py --data_path results/091719_datasets.csv --model_name ${model_name} --end_splits 5 --identifier 1208_add --exp_mode AddExp &> logs/120819_fimp_add.log &


## int: Run onlinenews (moving from int2 to int1, since flam only works in machine 1)
model_name='xgb-d1-o20-cv xgb-d3 rf-n1000 ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr rf-n1000 spline-v2'
counter=1
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name onlinenews \
        --model_name ${model_name} &> logs/120819_bias_var_onlinenews_${counter}.log &


# int: Run bias var tradeoff
counter=1
for model_name in 'ebm-bf-o50 ebm-o50-i50' 'ilr' 'lr' 'mlr' 'flam'; do
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/122119_bias_var_${counter}.log &
    echo $((counter++))
done


# TODO: Diabetes dies. Rerun diabetes in bias / var tradeoff??
model_name='spline-v2'
counter=3
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/120419_bias_var_${counter}.log &

# int2: Run removal and add for the auc metric
model_name='ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr skgbt-d1-o20-cv spline-v2 xgb-d1-o20-cv'
python feature_importances.py --data_path results/091719_datasets.csv --metric auc --model_name ${model_name} --end_splits 5 --identifier 1208_removal --exp_mode RemovalExp &> logs/122219_fimp_removal.log &
python feature_importances.py --data_path results/091719_datasets.csv --metric auc --model_name ${model_name} --end_splits 5 --identifier 1208_add --exp_mode AddExp &> logs/122219_fimp_add.log &


# int: Rerun bias/var tradeoff for the first job (it dies because of bad memory)
counter=1
model_name='ebm-bf-o50 ebm-o50-i50'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/122119_bias_var_${counter}.log &


## Rerun ilr, lr, mlr and ebm-o50-i50, ebm-bf-o50
d_name='onlinenews support2reg pneumonia'
counter=1
for model_name in 'ebm-bf-o50-q ilr-q' 'ebm-o50-i50-q lr mlr-q'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0108_quantile_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0108_quantile_summarize_${counter}.log) &
    echo $((counter++))
done

## int: Run other datasets to semi-simulate in different methods
counter=1
for model_name in 'ebm-bf-o50-q ilr-q' 'ebm-o50-i50-q lr mlr-q'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr spline-v2 flam \
        --d_name ${d_name} &> logs/0109_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0109_ss_summarize_${counter}.log) &
    echo $((counter++))
done

# Rerun everything with quantization
d_name='bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes onlinenews'
counter=1
for model_name in 'ebm-bf-o50-q ilr-q' 'ebm-o50-i50-q lr mlr-q'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0111_quantile_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0111_quantile_summarize_${counter}.log) &
    echo $((counter++))
done

# int2: Run the bias/var tradeoff for regression datasets!
counter=1
for model_name in 'ebm-bf-o50-q ilr-q' 'ebm-o50-i50-q lr mlr-q'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name onlinenews bikeshare calhousing support2reg wine breast churn pneumonia heart support2cls mimicii mimiciii adult credit diabetes
            --model_name ${model_name} &> logs/011319_bias_var_${counter}.log &
    echo $((counter++))
done

# int: rerun bias/var in flam with newer code to control for the memory
counter=5
model_name='flam'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name bikeshare breast calhousing churn pneumonia heart support2reg support2cls mimicii mimiciii wine adult credit diabetes \
        --model_name ${model_name} &> logs/122119_bias_var_${counter}.log &


###### Run more SS datasets
counter=1
for d_name in "
ss_mimicii_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-bf-o50 ss_mimicii_b1_r1377_xgb-d1-o20-cv ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2
ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2" \
"ss_breast_b1_r1377_ebm-o50-i50 ss_breast_b1_r1377_ebm-bf-o50 ss_breast_b1_r1377_xgb-d1-o20-cv ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2
ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam ss_adult_b1_r1377_spline-v2"; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam spline-v2 \
        --d_name ${d_name} &> logs/0127_ss_more_datasets_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0127_ss_more_datasets_summarize_${counter}.log) &
    echo $((counter++))
done

## Run credit and diabetes?
counter=1
for d_name in \
"ss_credit_b1_r1377_ebm-o50-i50 ss_credit_b1_r1377_ebm-bf-o50 ss_credit_b1_r1377_xgb-d1-o20-cv ss_credit_b1_r1377_lr ss_credit_b1_r1377_flam ss_credit_b1_r1377_spline-v2" \
"ss_diabetes_b1_r1377_ebm-o50-i50 ss_diabetes_b1_r1377_ebm-bf-o50 ss_diabetes_b1_r1377_xgb-d1-o20-cv ss_diabetes_b1_r1377_lr ss_diabetes_b1_r1377_flam ss_diabetes_b1_r1377_spline-v2" \
; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam spline-v2 \
        --d_name ${d_name} &> logs/0128_ss_more_datasets_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0128_ss_more_datasets_summarize_${counter}.log) &
    echo $((counter++))
done

# (waiting) int2: Rerun the ss since it dies when training spline-v2 in the dataset ss_adult_b1_r1377_flam
counter=2
d_name="ss_adult_b1_r1377_spline-v2"
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam spline-v2 \
    --d_name ${d_name} &> logs/0201_ss_more_datasets_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0201_ss_more_datasets_summarize_${counter}.log) &

# int: run the feature importances for using training to select and using test to report metric!
model_name='ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr skgbt-d1-o20-cv spline-v2 xgb-d1-o20-cv'
for metric in 'auc' 'mse'; do
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0201_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --end_splits 5  &> logs/020120_fimp_removal_${metric}.log &
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0201_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --end_splits 5  &> logs/020120_fimp_add_${metric}.log &
done

# (waiting) int2: run a shuffled dataset and compare them
model_name='ebm-o50-i50 xgb-d1-o20-cv flam spline-v2'
counter=1
for d_name in "ss_pneumonia_b1_r1377_ebm-o50-i50_sh ss_pneumonia_b1_r1377_xgb-d1-o20-cv_sh" "ss_pneumonia_b1_r1377_spline-v2_sh ss_pneumonia_b1_r1377_flam_sh"; do
    (python -u main.py --identifier 0201_ss_shuffle --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0201_ss_shuffle_${counter}.log && \
     python -u summarize.py --identifier 0201 \
        --data_path results/0201_ss_shuffle.csv --d_name ${d_name} --model_name ${model_name} &> logs/0201_ss_shuffle_summarize_${counter}.log) &
    echo $((counter++))
done

python -u summarize.py --identifier 0201 --overwrite 1 \
        --d_name ss_pneumonia_b1_r1377_ebm-o50-i50_sh ss_pneumonia_b1_r1377_xgb-d1-o20-cv_sh ss_pneumonia_b1_r1377_spline-v2_sh ss_pneumonia_b1_r1377_flam_sh \
        --data_path results/0201_ss_shuffle.csv --model_name gnd_truth

# (has not run) bias var on ss with diff generators
counter=1
for d_name in 'ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv'; do
    python -u main.py --identifier 020120_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name ebm-o50-i50 xgb-d1-o20-cv &> logs/020120_ss_bias_var_d${counter}.log &
    echo $((counter++))
done

# bias/var on ss with diff generators
counter=1
for d_name in 'ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50'; do
    python -u main.py --identifier 020120_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name ebm-o50-i50 ebm-bf-o50 &> logs/020120_ss_bias_var_d${counter}.log &
    echo $((counter++))
done


########## Run new model spline-b ################
# 1) int: run the normal datasets (skip the onlinenews, diabetes dataset)
model_name='spline-b'
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia heart adult' 'support2reg support2cls mimicii mimiciii wine credit'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0202_spline-b_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} \
        &> logs/0202_spline-b_normal_summarize_${counter}.log) &
    echo $((counter++))
done

model_name='spline-b'
counter=3
d_name='onlinenews diabetes'
(python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
    --model_name ${model_name} \
    --d_name ${d_name} &> logs/0202_spline-b_normal_datasets_${counter}.log && \
    python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} \
    &> logs/0202_spline-b_normal_summarize_${counter}.log) &


# 1.5) run the rspline on all old ss datasets
counter=1
for d_name in "ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam" \
"ss_churn_b1_r1377_ebm-o50-i50 ss_churn_b1_r1377_ebm-bf-o50 ss_churn_b1_r1377_xgb-d1-o20-cv ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam
ss_support2cls_b1_r1377_ebm-o50-i50 ss_support2cls_b1_r1377_ebm-bf-o50 ss_support2cls_b1_r1377_xgb-d1-o20-cv ss_support2cls_b1_r1377_lr ss_support2cls_b1_r1377_flam"; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name spline-b \
        --d_name ${d_name} &> logs/0202_ss_splineb_oldss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0202_ss_splineb_oldss_summarize_${counter}.log) &
    echo $((counter++))
done

counter=1
for d_name in "ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam" \
"ss_churn_b1_r1377_ebm-o50-i50 ss_churn_b1_r1377_ebm-bf-o50 ss_churn_b1_r1377_xgb-d1-o20-cv ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam
ss_support2cls_b1_r1377_ebm-o50-i50 ss_support2cls_b1_r1377_ebm-bf-o50 ss_support2cls_b1_r1377_xgb-d1-o20-cv ss_support2cls_b1_r1377_lr ss_support2cls_b1_r1377_flam"; do
    (python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0123_ss_rspline_oldss_summarize_${counter}.log) &
    echo $((counter++))
done

# Rerun the summarization due to the factor [1, 2] bug
counter=1
d_name="ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam"
    (python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0202_ss_splineb_oldss_summarize_${counter}.log) &



# 2) run the spline datasets
model_name='spline-b'
counter=1
for d_name in 'ss_pneumonia_b1_r1377_spline-b ss_mimiciii_b1_r1377_spline-b' 'ss_churn_b1_r1377_spline-b ss_support2cls_b1_r1377_spline-b'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam spline-b \
        --d_name ${d_name} &> logs/0202_ss_splineb_ownss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} --model_name ${model_name} &> logs/0202_ss_splineb_ownss_summarize_${counter}.log) &
    echo $((counter++))
done

# rerun ->
model_name='spline-b'
counter=1
d_name='ss_pneumonia_b1_r1377_spline-b ss_mimiciii_b1_r1377_spline-b'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam spline-b \
        --d_name ${d_name} &> logs/0202_ss_splineb_ownss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} --model_name ${model_name} &> logs/0202_ss_splineb_ownss_summarize_${counter}.log) &


# 3) run the bias/var tradeoff
# TODO: remove onlinenews. check the result
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia adult heart' 'support2reg support2cls mimicii mimiciii wine credit diabetes onlinenews'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name rspline &> logs/012319_rspline_bias_var_d${counter}.log &
    echo $((counter++))
done
# 4) Run removal / add experiment for these models


# (running) int2: Keep debuging rspline models??
model_name='rspline-k10-s rspline-k10 rspline-k200 rspline-k200-s'
counter=1
for d_name in 'breast pneumonia'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0203_rspline_debug_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0203_rspline_debug_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# (running) set different ratio of shuffling the datasets
model_name='xgb-d1-o20-cv'
counter=1
for d_name in "ss_pneumonia_b1_r1377_xgb-d1-o20-cv_sh0.2 ss_pneumonia_b1_r1377_xgb-d1-o20-cv_sh0.5 ss_pneumonia_b1_r1377_xgb-d1-o20-cv_sh0.8"; do
    (python -u main.py --identifier 0201_ss_shuffle --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0203_ss_shuffle_${counter}.log && \
     python -u summarize.py --identifier 0201 \
        --data_path results/0201_ss_shuffle.csv --d_name ${d_name} --model_name ${model_name} &> logs/0203_ss_shuffle_summarize_${counter}.log) &
    echo $((counter++))
done

# (running)) int: run the feature importances for using test to select and using test to report metric!
# ignore skgbt-d1-o20-cv, spline-b for now!
model_name='ebm-bf-o50 ebm-o50-i50 flam ilr lr mlr xgb-d1-o20-cv'
d_name='churn heart calhousing mimicii mimiciii pneumonia adult breast support2cls support2reg wine bikeshare diabetes credit onlinenews'
for metric in 'auc' 'mse'; do
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020320_fimp_removal_${metric}.log &
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020320_fimp_add_${metric}.log &
done



########## Run new model rspline ################
# 1) run the normal datasets (skip the onlinenews dataset) # As usual onlinenews fails
model_name='rspline-v2'
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia adult heart' 'support2reg support2cls mimicii mimiciii wine credit diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0204_rspline_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0204_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done

d_name='bikeshare breast calhousing churn pneumonia adult heart'
counter=1
model_name='rspline-v2'
python -u summarize.py --identifier 0927 --data_path results/091719_datasets.csv \
--d_name ${d_name} &> logs/0204_rspline_normal_summarize_${counter}.log &


# calhousing dies
for d_name in 'bikeshare' 'breast' 'calhousing' 'churn' 'pneumonia' 'adult' 'heart' 'support2reg' 'support2cls' 'mimicii' 'mimiciii' 'wine' 'credit' 'diabetes'; do
    python testing.py --d_name ${d_name} &
done

# resummarize
model_name='rspline'
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia adult heart' 'support2reg support2cls mimicii mimiciii wine credit diabetes'; do
    (python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0123_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# rerun the whole models with overwrite as 1
model_name='rspline'
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia adult heart' 'support2reg support2cls mimicii mimiciii wine credit diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 1 \
        --d_name ${d_name} &> logs/0123_rspline_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0123_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# Run the normal GAM and see the performance and hopefully it won't flucutate
model_name='rspline-gam'
counter=1
for d_name in 'pneumonia churn bikeshare breast calhousing adult heart' 'mimicii mimiciii support2cls support2reg wine credit diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0123_rspline_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0123_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# reduce the maxk...
model_name='rspline-gam-k50'
counter=1
for d_name in 'pneumonia churn bikeshare breast calhousing adult heart' 'mimicii mimiciii support2cls support2reg wine credit diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0126_rspline_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0126_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# reduce the maxk...
model_name='rspline-k'
counter=1
for d_name in 'pneumonia churn bikeshare breast calhousing adult heart' 'mimicii mimiciii support2cls support2reg wine credit diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0126_rspline_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0126_rspline_normal_summarize_${counter}.log) &
    echo $((counter++))
done



# 1.5) run the rspline on all old ss datasets
d_name='ss_pneumonia_b1_r1377_ebm-o50-i50'
model_name='rspline-v2'
counter=999
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0204_ss_rspline_oldss_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0204_ss_rspline_oldss_summarize_${counter}.log) &

counter=1
model_name='rspline-v2'
for d_name in \
"ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam" \
"ss_mimicii_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-bf-o50 ss_mimicii_b1_r1377_xgb-d1-o20-cv ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam
ss_breast_b1_r1377_ebm-o50-i50 ss_breast_b1_r1377_ebm-bf-o50 ss_breast_b1_r1377_xgb-d1-o20-cv ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam
ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam" \
"ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam" \
"ss_churn_b1_r1377_ebm-o50-i50 ss_churn_b1_r1377_ebm-bf-o50 ss_churn_b1_r1377_xgb-d1-o20-cv ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam
ss_support2cls_b1_r1377_ebm-o50-i50 ss_support2cls_b1_r1377_ebm-bf-o50 ss_support2cls_b1_r1377_xgb-d1-o20-cv ss_support2cls_b1_r1377_lr ss_support2cls_b1_r1377_flam"; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0204_ss_rspline_oldss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0204_ss_rspline_oldss_summarize_${counter}.log) &
    echo $((counter++))
done

# resummarize rspline-v2 => gets wierd graph mse
counter=1
model_name='rspline-v2'
for d_name in \
"ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam" \
"ss_mimicii_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-bf-o50 ss_mimicii_b1_r1377_xgb-d1-o20-cv ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam
ss_breast_b1_r1377_ebm-o50-i50 ss_breast_b1_r1377_ebm-bf-o50 ss_breast_b1_r1377_xgb-d1-o20-cv ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam
ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam" \
"ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam" \
"ss_churn_b1_r1377_ebm-o50-i50 ss_churn_b1_r1377_ebm-bf-o50 ss_churn_b1_r1377_xgb-d1-o20-cv ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam
ss_support2cls_b1_r1377_ebm-o50-i50 ss_support2cls_b1_r1377_ebm-bf-o50 ss_support2cls_b1_r1377_xgb-d1-o20-cv ss_support2cls_b1_r1377_lr ss_support2cls_b1_r1377_flam"; do
    (python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} --model_name ${model_name} --overwrite 1 &> logs/0206_ss_rspline_oldss_summarize_${counter}.log) &
    echo $((counter++))
done

# run a rspline-s on breast and adult and see if it also has a big wierd phenonemon
counter=1
model_name='rspline-v2-s'
for d_name in \
"ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam" \
"ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam" \
; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0207_ss_rsplines_oldss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0207_ss_rsplines_oldss_summarize_${counter}.log) &
    echo $((counter++))
done


# 2) run the ss rspline datasets
# TODO: run a spline-v2 instead of spline_v2
counter=1
model_name='ebm-o50-i50 ebm-bf-o50 xgb-d1-o20-cv mlr lr ilr flam rspline-v2 spline-v2'
for d_name in \
'ss_pneumonia_b1_r1377_rspline-v2 ss_churn_b1_r1377_rspline-v2 ss_support2cls_b1_r1377_rspline-v2' \
'ss_mimicii_b1_r1377_rspline-v2 ss_breast_b1_r1377_rspline-v2' 'ss_mimiciii_b1_r1377_rspline-v2 ss_heart_b1_r1377_rspline-v2' 'ss_adult_b1_r1377_rspline-v2'\
; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0204_ss_ownss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0204_ss_ownss_summarize_${counter}.log) &
    echo $((counter++))
done


# 3) run the bias/var tradeoff
# TODO: remove onlinenews. check the result
counter=1
for d_name in 'bikeshare breast calhousing churn pneumonia heart credit' 'support2reg support2cls mimicii mimiciii wine diabetes adult'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name rspline-v2 &> logs/020419_rspline_bias_var_d${counter}.log &
    echo $((counter++))
done
# 4) Run removal / add experiment for these models
model_name='rspline-v2'
metric='mse'
python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_rspline_fimp_removal_${metric}.log &
python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_rspline_fimp_add_${metric}.log &

############ Conclusion: Check rspline performance

## TODO: l1 to l2 ratio of XGB. Showing different graphs and 
# run xgb with different cols 1.0 0.5 0.1 0.00001
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.75 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.25 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1'
counter=1
# for d_name in 'bikeshare breast calhousing churn pneumonia adult heart diabetes' 'support2reg support2cls mimicii mimiciii wine credit'; do
for d_name in 'pneumonia' 'mimiciii'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0205_xgbl1l2_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0205_xgbl1l2_normal_summarize_${counter}.log) &
    echo $((counter++))
done
model_name='rspline-v2-s'
counter=1
# for d_name in 'bikeshare breast calhousing churn pneumonia adult heart diabetes' 'support2reg support2cls mimicii mimiciii wine credit'; do
for d_name in 'pneumonia mimiciii'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0205_xgbl1l2_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0205_xgbl1l2_normal_summarize_${counter}.log) &
    echo $((counter++))
done
####### (wait) run an add/removal experiment for the mse on these l1 / l2
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.75 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.25 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s rspline-v2'
counter=1
for d_name in 'pneumonia mimiciii'; do
    for metric in 'mse' 'logloss'; do
        python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020420_l1l2_fimp_removal_${metric}.log &
        python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020420_l1l2_fimp_add_${metric}.log &
    done
done

## Run the ebm-bf of the shuffling dataset
model_name='ebm-bf-o50'
counter=1
for d_name in "ss_pneumonia_b1_r1377_ebm-bf-o50_sh ss_pneumonia_b1_r1377_ebm-bf-o50_sh0.2 ss_pneumonia_b1_r1377_ebm-bf-o50_sh0.5 ss_pneumonia_b1_r1377_ebm-bf-o50_sh0.8"; do
    (python -u main.py --identifier 0201_ss_shuffle --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0203_ss_shuffle_${counter}.log && \
     python -u summarize.py --identifier 0201 \
        --data_path results/0201_ss_shuffle.csv --d_name ${d_name} --model_name ${model_name} &> logs/0203_ss_shuffle_summarize_${counter}.log) &
    echo $((counter++))
done
## Run the ebm of the shuffling dataset
model_name='ebm-o50-i50'
counter=1
for d_name in "ss_pneumonia_b1_r1377_ebm-o50-i50_sh ss_pneumonia_b1_r1377_ebm-o50-i50_sh0.2 ss_pneumonia_b1_r1377_ebm-o50-i50_sh0.5 ss_pneumonia_b1_r1377_ebm-o50-i50_sh0.8"; do
    (python -u main.py --identifier 0201_ss_shuffle --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0204_ss_shuffle_${counter}.log && \
     python -u summarize.py --identifier 0201 \
        --data_path results/0201_ss_shuffle.csv --d_name ${d_name} --model_name ${model_name} &> logs/0204_ss_shuffle_summarize_${counter}.log) &
    echo $((counter++))
done


## int2: run an add/removal with metric logloss
d_name='bikeshare breast calhousing churn pneumonia heart credit support2reg support2cls mimicii mimiciii wine diabetes adult'
model_name='ebm-bf-o50-q ebm-o50-i50-q ilr-q mlr-q lr xgb-d1-o20-cv flam rspline-v2'
counter=1
for metric in 'logloss'; do
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_logloss_fimp_removal_${metric}.log &
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_logloss_fimp_add_${metric}.log &
done



#### spline-b is not as good as spline-v2, with overall slightly worse performance. Give up.

# More L1/L2 models on every dataset!
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'
counter=1
for d_name in 'bikeshare breast' 'adult' 'calhousing churn heart' 'diabetes'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0205_l1l2_more_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0205_l1l2_more_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_logloss_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_logloss_${counter}.log
    ) &
    echo $((counter++))
done
# second round on int2: runnning more datasets
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'
counter=5
for d_name in 'support2reg support2cls' 'mimicii wine' 'credit'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0205_l1l2_more_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0205_l1l2_more_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_logloss_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_logloss_${counter}.log
    ) &
    echo $((counter++))
done

# rerun since the lr-l1 dies in regression case :(
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'

# counter=1
# d_name='breast'
# counter=3
# d_name='churn heart'
# counter=5
# d_name='support2cls'
counter=4
d_name='diabetes'
# counter=6
# d_name='mimicii'
(python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
    --model_name ${model_name} \
    --d_name ${d_name} &> logs/0205_l1l2_more_normal_datasets_${counter}.log && \
    python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0205_l1l2_more_normal_summarize_${counter}.log && \
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_mse_${counter}.log && \
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_mse_${counter}.log && \
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_removal_logloss_${counter}.log && \
    python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_l1l2_fimp_add_logloss_${counter}.log
) &

# Miss the feat imp for rspline-v2 for following datasets
d_name='mimicii support2cls support2reg wine credit'
model_name='rspline-v2'
metric='mse'
python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_rspline_fimp_rest_removal_${metric}.log &
python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric ${metric} --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020520_rspline_fimp_rest_add_${metric}.log &

###### Bias/Var for these l1/l2 models (only cls dataset)
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'
counter=1 # for int2
for d_name in 'support2cls mimicii wine' 'credit'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name ${model_name} &> logs/020619_l1l2_bias_var_d${counter}.log &
    echo $((counter++))
done
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'
counter=3 # for int
for d_name in 'breast adult calhousing churn heart' 'diabetes'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name ${model_name} &> logs/020619_l1l2_bias_var_d${counter}.log &
    echo $((counter++))
done
# Make 3 into 5, 6 in int2 to run
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1 rspline-v2-s'
counter=5 # for int
for d_name in 'breast calhousing' 'churn heart adult'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name ${d_name} \
            --model_name ${model_name} &> logs/020619_l1l2_bias_var_d${counter}.log &
    echo $((counter++))
done
model_name='xgb-d1-o20-cols1 xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1 xgb-d1-o20-cols0.00001 lr-l1'
d_name='breast'
counter=5
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
    --d_name ${d_name} \
    --model_name ${model_name} &> logs/020619_l1l2_bias_var_d${counter}.log &


# run the compass dataset: (ignore xgb-d1-o20-cv) to make sure other finish earlier!
# Have not run these model_name 
d_name='compass'
# counter=1
# for model_name in 'ebm-o50-i50-q ebm-bf-o50-q' 'xgb-d1-o20-cols0.00001 xgb-d1-o20-cols1' 'xgb-d1-o20-cv'; do
# counter=4
# for model_name in 'xgb-d1-o20-cols0.5 xgb-d1-o20-cols0.1' 'rspline-v2 rspline-v2-s' 'lr lr-l1 mlr-q ilr-q flam'; do
# counter=6
# for model_name in 'xgb-d1-o20-cv'; do
# counter=7
# for model_name in 'rf-n1000 xgb-d3 xgb-d3-o20'; do
# rspline-v2 performs much worse in the ss. Change back to python spline. Run compass with spline-v2
counter=8
for model_name in 'spline-v2' 'spline-b'; do
    (\
     python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0206_compass_more_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0206_compass_more_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020620_compass_fimp_removal_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric mse --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020620_compass_fimp_add_mse_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_removal --exp_mode RemovalExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020620_compass_fimp_removal_logloss_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020620_compass_fimp_add_logloss_${counter}.log && \
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/020619_compass_bias_var_d${counter}.log && \
     echo
    ) &
    echo $((counter++))
done

###### forget to run some models with all the adding with logloss
d_name='adult breast churn compass credit diabetes heart mimicii mimiciii pneumonia support2cls'
counter=1
for model_name in 'spline-v2' 'xgb-d1-o20-cols0.00001 xgb-d1-o20-cols0.1' 'xgb-d1-o20-cols0.5 xgb-d1-o20-cols1' 'lr-l1'; do
    (\
     python feature_importances.py --data_path results/091719_datasets.csv --identifier 0203_add --exp_mode AddExp --metric logloss --model_name ${model_name} --d_name ${d_name} --end_splits 5  &> logs/020720_fimp_add_logloss_${counter}.log && \
     echo
    ) &
    echo $((counter++))
done



# Resummarize
counter=1
model_name='gnd_truth'
for d_name in \
"ss_pneumonia_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_ebm-bf-o50 ss_pneumonia_b1_r1377_xgb-d1-o20-cv ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam 
ss_mimiciii_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-bf-o50 ss_mimiciii_b1_r1377_xgb-d1-o20-cv ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam 
ss_mimicii_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-bf-o50 ss_mimicii_b1_r1377_xgb-d1-o20-cv ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam 
ss_breast_b1_r1377_ebm-o50-i50 ss_breast_b1_r1377_ebm-bf-o50 ss_breast_b1_r1377_xgb-d1-o20-cv ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam
ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam
ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam 
ss_churn_b1_r1377_ebm-o50-i50 ss_churn_b1_r1377_ebm-bf-o50 ss_churn_b1_r1377_xgb-d1-o20-cv ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam 
ss_support2cls_b1_r1377_ebm-o50-i50 ss_support2cls_b1_r1377_ebm-bf-o50 ss_support2cls_b1_r1377_xgb-d1-o20-cv ss_support2cls_b1_r1377_lr ss_support2cls_b1_r1377_flam"; do
    (python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} --model_name ${model_name} --overwrite 1 &> logs/0207_ss_gnd_truth_resummarize_${counter}.log) &
    echo $((counter++))
done


######################### One hot encodingggggggggggg #################
# adult churn compass diabetes heart support2cls (support2reg)

##### Now only for running these new datasets
# 1) re-running these models on adult churn compass heart
model_name='xgb-l-d1-o20-cols0.5 ebm-o50-i50-q ebm-bf-o50-q rf-n1000 ilr-q mlr-q lr'
counter=1
for d_name in 'adult' 'compass' 'churn heart'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log) &
    echo $((counter++))
done

model_name='xgb-l-d1-o20-cols0.5 ebm-o50-i50-q ebm-bf-o50-q rf-n1000 ilr-q mlr-q'
counter=1
for d_name in 'adult' 'churn heart'; do
    (python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log) &
    echo $((counter++))
done

counter=3
d_name='churn heart'
python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log &

model_name='spline-v2 lr'
counter=10
for d_name in 'adult' 'compass' 'churn heart'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log) &
    echo $((counter++))
done


# 1.2) (int2) run lots of l1/l2 models on compass
model_name='xgb-l-d1-o20-cols0.1 xgb-l-d1-o20-cols0.00001 xgb-l-d1-o20-cols1'
counter=4
for d_name in 'compass'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log) &
    echo $((counter++))
done

# 1.1) run on new datasets support2cls2, diabetes2
d_name='support2cls2 diabetes2 support2reg2'
counter=5
for model_name in 'xgb-l-d1-o20-cols0.5 ebm-o50-i50-q ebm-bf-o50-q rf-n1000 ilr-q mlr-q lr' 'spline-v2' 'flam'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log) &
    echo $((counter++))
done

counter=5
model_name='xgb-l-d1-o20-cols0.5 ebm-o50-i50-q ebm-bf-o50-q rf-n1000 ilr-q mlr-q lr'
# counter=6
# model_name='spline-v2'
d_name='support2cls2 diabetes2 support2reg2'
python -u summarize.py --identifier 0927 --overwrite 1 \
    --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log &



# testing the summarizatino on xgb-d1-o20
python -u summarize.py --identifier 0927 --overwrite 1 \
    --data_path results/091719_datasets.csv --d_name adult --model_name xgb-d1-o20


model_name='lr-l1 flam xgb-l-d1-o20-cols0.00001 xgb-l-d1-o20-cols0.1 xgb-l-d1-o20-cols0.5 xgb-l-d1-o20-cols1 ebm-bf-o50-q ebm-o50-i50-q ilr-q mlr-q spline-v2 lr'
counter=1
for d_name in 'adult' 'compass' 'churn' 'heart'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

############## 2) Run the feature importances
# rerun other datasets
d_name='compass adult churn'
counter=1
for model_name in 'xgb-l-d1-o20-cols0.1' 'xgb-l-d1-o20-cols0.00001' 'xgb-l-d1-o20-cols0.5' 'xgb-l-d1-o20-cols1' 'ebm-bf-o50-q' 'ebm-o50-i50-'q' spline-v2' 'lr' 'ilr-q' 'mlr-q' 'lr-l1' 'flam'; do
    python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log &
done


#### For test set performance:
model_name='xgb-d1-o20-cols0.5'
counter=15
for d_name in 'diabetes2 support2cls2'; do
    (python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

#### For test set performance to do the ebm with onehot encoding lol
counter=20
model_name='ebm-h-bf-o50-q ebm-h-o50-i50-q'
for d_name in 'adult churn compass' 'heart diabetes2 support2cls2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

# TODO: Find a bug; rerun all the label encoders, but use the trained model.
counter=30
model_name='mlr-q ilr-q rf-n1000 ebm-bf-o50-q ebm-o50-i50-q xgb-l-d1-o20-cols0.5'
for d_name in 'adult churn compass' 'heart diabetes2 support2cls2 support2reg2'; do
    (python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done


#### bias / var
model_name='mlr-q ebm-o50-i50-q xgb-l-d1-o20-cols0.5'
counter=1
for d_name in 'adult churn compass' 'heart diabetes2 support2cls2 support2reg2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

counter=3
model_name='ilr-q rf-n1000 ebm-bf-o50-q'
for d_name in 'adult churn compass' 'heart diabetes2 support2cls2 support2reg2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

# Run the  on labeled datasets
counter=40
model_name='xgb-l-d3-cols0.5 xgb-l-d3-o20-cols0.5'
for d_name in 'adult churn compass' 'heart diabetes2 support2cls2 support2reg2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=50
model_name='spline-v2'
d_name='diabetes2'
(python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
    --model_name ${model_name} --overwrite 0 \
    --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
    python -u summarize.py --identifier 0927 --overwrite 1 \
    --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
    python feature_importances.py --data_path results/091719_datasets.csv \
    --identifier 0210_add --exp_mode AddExp --metric logloss \
    --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
    &> logs/021020_compass_fimp_add_logloss_${counter}.log \
) &
echo $((counter++))

# Run the bias/var for these
# - Run xgb-d1-o20-cols0.5 for diabetes2, mimiciii, pneumonia, support2cls2
model_name='xgb-d1-o20-cols0.5'
counter=10
for d_name in 'diabetes2 mimiciii pneumonia support2cls2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done
# - Run spline-v2, lr on diabetes2 and support2cls2
model_name='spline-v2 lr'
counter=20
for d_name in 'support2cls2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done
# - run on 
model_name='xgb-d1-o20-cols0.00001'
counter=30
for d_name in 'diabetes2 mimiciii pneumonia support2cls2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

# bias/var
model_name='xgb-d1-o20-cols1'
counter=50
for d_name in 'diabetes2 mimiciii' 'pneumonia support2cls2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

### TORUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
counter=90
model_name='xgb-l-d1-o20-cols1'
for d_name in 'churn compass heart support2cls2 adult'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

counter=70
model_name='flam'
for d_name in 'support2cls2 diabetes2'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done

# SRUN VERSION!!!!
counter=90
model_name='xgb-l-d1-o20-cols1'
for d_name in 'churn' 'compass' 'heart support2cls2' 'adult'; do
    ./srun.sh -o logs/021019_enc_bias_var_d${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &
    echo $((counter++))
done

# - Run spline-v2, lr on diabetes2 and support2cls2

d_name='support2cls2'
counter=20
for model_name in 'spline-v2'; do
    srun --mem=6G --gres=gpu:0 -c 15 -l -p p100 -o logs/021019_enc_bias_var_d${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &
    echo $((counter++))
done

# - run on 
model_name='xgb-d1-o20-cols0.00001'
counter=30
for d_name in 'mimiciii' 'pneumonia' 'support2cls2'; do
    ./srun.sh -o logs/021019_enc_bias_var_d${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &
    echo $((counter++))
done
# pneumonia int
counter=31
model_name='xgb-d1-o20-cols0.00001'
d_name='pneumonia'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
# bias/var

model_name='xgb-d1-o20-cols1'
counter=50
for d_name in 'mimiciii' 'support2cls2'; do
     ./srun.sh -o logs/021019_enc_bias_var_d${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &
    echo $((counter++))
done
counter=52
model_name='xgb-d1-o20-cols1'
d_name='pneumonia'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
### TORUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
counter=90
model_name='xgb-l-d1-o20-cols1'
for d_name in 'churn compass heart support2cls2 adult'; do
     python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 --d_name ${d_name} --model_name ${model_name} &> logs/021019_enc_bias_var_d${counter}.log &
    echo $((counter++))
done


# XGB with label encoding and column subsample as 1!!
counter=80
model_name='xgb-l-d1-o20-cols1'
for d_name in 'adult churn compass' 'heart support2cls2 diabetes2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

# Couple holes in the generlaization dataset
counter=90
d_name='support2cls2'
for model_name in 'xgb-d1-o20-cols1' 'xgb-d3 xgb-d3-o20'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=100
model_name='xgb-d1'
for d_name in 'adult breast churn credit heart mimicii mimiciii pneumonia diabetes2 compass support2cls2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
     python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
     python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done



# Run on these old datasets with xgb ilr mlr

counter=1
model_name='xgb-d1-o20-cols1 ilr-q mlr-q'
for d_name in \
'ss_pneumonia_b1_r1377_ebm-o50-i50 ss_mimiciii_b1_r1377_ebm-o50-i50 ss_pneumonia_b1_r1377_spline-v2 ss_mimiciii_b1_r1377_spline-v2 
ss_pneumonia_b1_r1377_flam ss_mimiciii_b1_r1377_flam ss_pneumonia_b1_r1377_lr ss_mimiciii_b1_r1377_lr ss_churn_b1_r1377_spline-v2 
ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam ss_breast_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-o50-i50' \
'ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2 
ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2 ss_breast_b1_r1377_lr 
ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2 ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam ss_adult_b1_r1377_spline-v2'; do
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
    --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
    python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
echo $((counter++))
done
counter=2
model_name='xgb-d1-o20-cols1 ilr-q mlr-q'
d_name='ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2 
ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2 ss_breast_b1_r1377_lr 
ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2 ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam ss_adult_b1_r1377_spline-v2'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
    --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
    python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &

# xgb on these pneumonia mimiciii mimicii churn breast heart adult

counter=10
model_name='xgb-d1-o20-cols1 ilr-q mlr-q ebm-o50-i50-q ebm-bf-o50 lr flam spline-v2'
for d_name in \
'ss_adult_b1_r1377_xgb-d1-o20-cols1 ss_churn_b1_r1377_xgb-d1-o20-cols1 ss_heart_b1_r1377_xgb-d1-o20-cols1' \
'ss_pneumonia_b1_r1377_xgb-d1-o20-cols1 ss_mimiciii_b1_r1377_xgb-d1-o20-cols1 ss_mimicii_b1_r1377_xgb-d1-o20-cols1' \
'ss_breast_b1_r1377_xgb-d1-o20-cols1 ss_adult_b1_r1377_ebm-o50-i50-q' \
'ss_churn_b1_r1377_ebm-o50-i50-q ss_heart_b1_r1377_ebm-o50-i50-q'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
    echo $((counter++))
done

counter=160
for model_name in 'spline-v2'; do
for d_name in \
'ss_adult_b1_r1377_xgb-d1-o20-cols1' 'ss_churn_b1_r1377_xgb-d1-o20-cols1' 'ss_heart_b1_r1377_xgb-d1-o20-cols1' \
'ss_pneumonia_b1_r1377_xgb-d1-o20-cols1' 'ss_mimiciii_b1_r1377_xgb-d1-o20-cols1' 'ss_mimicii_b1_r1377_xgb-d1-o20-cols1' \
'ss_breast_b1_r1377_xgb-d1-o20-cols1' 'ss_adult_b1_r1377_ebm-o50-i50-q'; do
    (./srun.sh -o logs/0211_ss_${counter}.log python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} && \
     ./srun.sh -python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
    echo $((counter++))
done
done

counter=220
model_name='ebm-bf-o50-q'
for d_name in 'ss_churn_b1_r1377_flam' 'ss_churn_b1_r1377_lr' 'ss_churn_b1_r1377_spline-v2' 'ss_heart_b1_r1377_flam' 'ss_heart_b1_r1377_lr' 'ss_heart_b1_r1377_spline-v2'; do
    (./srun.sh -o logs/0211_ss_${counter}.log python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name}) &
    echo $((counter++))
done

d_name='ss_adult_b1_r1377_lr'
python -u summarize.py --identifier 1112 --overwrite 1 \
    --data_path results/1108_ss.csv --d_name ${d_name}


counter=160
model_name='spline-v2'
for d_name in 'ss_adult_b1_r1377_xgb-d1-o20-cols1' 'ss_adult_b1_r1377_ebm-o50-i50-q'; do
(./srun.sh -o logs/0211_ss_${counter}.log python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name}) &
echo $((counter++))
done

#  
counter=180
for model_name in 'xgb-d1-o20-cols1 ebm-o50-i50-q ebm-bf-o50 ilr-q mlr-q lr' 'flam'; do
for d_name in \
'ss_adult_b1_r1377_xgb-d1-o20-cols1' 'ss_churn_b1_r1377_xgb-d1-o20-cols1 ss_heart_b1_r1377_xgb-d1-o20-cols1' \
'ss_pneumonia_b1_r1377_xgb-d1-o20-cols1 ss_mimiciii_b1_r1377_xgb-d1-o20-cols1' 'ss_mimicii_b1_r1377_xgb-d1-o20-cols1' \
'ss_breast_b1_r1377_xgb-d1-o20-cols1 ss_adult_b1_r1377_ebm-o50-i50-q'; do
    (./srun.sh -o logs/0211_ss_${counter}.log python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} && \
     ./srun.sh -python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
    echo $((counter++))
done
done

## How do I do summarization Orz
d_name=\
'ss_churn_b1_r1377_xgb-d1-o20-cols1 ss_hepart_b1_r1377_xgb-d1-o20-cols1
ss_pneumonia_b1_r1377_xgb-d1-o20-cols1 ss_mimiciii_b1_r1377_xgb-d1-o20-cols1 
ss_mimicii_b1_r1377_xgb-d1-o20-cols1
ss_breast_b1_r1377_xgb-d1-o20-cols1
ss_churn_b1_r1377_ebm-o50-i50-q ss_heart_b1_r1377_ebm-o50-i50-q
' \
python -u summarize.py --identifier 1112 --overwrite 1 \
    --data_path results/1108_ss.csv --d_name ${d_name}

python -u summarize.py --identifier 1112 --overwrite 1 \
    --data_path results/1108_ss.csv --model_name ilr-q mlr-q xgb-d1-o20-cols1
python -u summarize.py --identifier 1112 --overwrite 1 \
    --data_path results/1108_ss.csv --model_name ebm-o50-i50-q ebm-bf-o50-q

d_name='\
ss_heart_b1_r1377_flam
ss_heart_b1_r1377_lr
ss_heart_b1_r1377_spline-v2
'
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} --model_name lr spline-v2 flam gnd_truth --overwrite 1





# int2" adult, heart and churn for all the ebm-o50-i50-q ebm-bf-o50-q
model_name='ebm-o50-i50-q'
d_name='ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2  
ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam ss_churn_b1_r1377_spline-v2'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &

model_name='ebm-o50-i50-q'
d_name='ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2  
ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam ss_churn_b1_r1377_spline-v2'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &

counter=210
d_name='ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam ss_adult_b1_r1377_spline-v2'
for model_name in 'ebm-o50-i50-q ebm-bf-o50-q'; do
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
done

# just for visualization...
counter=190
model_name='xgb-d1-o20-cols1 ebm-o50-i50-q ebm-bf-o50 flam spline-v2'
d_name='ss_compass_b1_r1377_spline-v2'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
python -u summarize.py --identifier 1112 \
    --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
echo $((counter++))

for d_name in \
'ss_adult_b1_r1377_xgb-d1-o20-cols1 ss_churn_b1_r1377_xgb-d1-o20-cols1 ss_heart_b1_r1377_xgb-d1-o20-cols1' \
'ss_pneumonia_b1_r1377_xgb-d1-o20-cols1 ss_mimiciii_b1_r1377_xgb-d1-o20-cols1 ss_mimicii_b1_r1377_xgb-d1-o20-cols1' \
'ss_breast_b1_r1377_xgb-d1-o20-cols1 ss_adult_b1_r1377_ebm-o50-i50-q' \
'ss_churn_b1_r1377_ebm-o50-i50-q ss_heart_b1_r1377_ebm-o50-i50-q'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &
    echo $((counter++))
done


counter=13
model_name='xgb-d1-o20-cols1 ilr-q mlr-q ebm-o50-i50-q ebm-bf-o50 lr flam spline-v2'
d_name='ss_churn_b1_r1377_ebm-o50-i50-q ss_heart_b1_r1377_ebm-o50-i50-q'
(python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0211_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0211_ss_summarize_${counter}.log) &

# EBM-BF / EBM on different encodings

####### SS datasets. What generators to run????????????
# adult churn heart support2cls2: at least I can run EBM with label encoding
# - rerun ss on ilr / mlr / ebm



# TODO: edit the 091719_datasets to replicate xgb-d1-o20 -> xgb-l-d1-o20 on datasets that does not have onehot encoding


# run ebm-h-bf-o50-q on 'adult churn compass diabetes2 heart support2cls2' 
counter=60
model_name='ebm-h-bf-o50-q'
for d_name in 'adult churn compass diabetes2 heart support2cls2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=60
model_name='ebm-h-bf-o50-q'
for d_name in 'adult churn compass diabetes2 heart support2cls2'; do
    (python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

# Run a one-hot encoding with col-subsample as 1
model_name='xgb-d1-o20-cols1'
counter=70
for d_name in 'diabetes2 support2cls2'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} &> logs/0209_encoding_normal_summarize_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=140
d_name='support2cls2'
for model_name in 'xgb-l-d1-o20-cols0.00001'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=130
model_name='xgb-d1-o20-cols1'
for d_name in 'adult' 'churn' 'breast'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=140
d_name='support2cls2'
for model_name in 'xgb-l-d1-o20-cols0.00001'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

counter=150
d_name='support2cls2 adult churn heart compass'
for model_name in 'xgb-d1-o20-cols0.00001'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 0 \
        --d_name ${d_name} &> logs/0209_encoding_normal_datasets_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/021020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done




# TODO: find a new regression dataset?
# Problem => this is a bit unfair to all those binning methods


# Optional TODO: Run the SS with quantization? It seems the cls dataset does not change. So maybe not...
# Optional TODO: Run the bias/var in classification with quantization (wait for a long holiday)
counter=3
for model_name in 'ebm-bf-o50-q ilr-q ebm-o50-i50-q lr mlr-q'; do
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
            --d_name breast churn pneumonia heart support2cls mimicii mimiciii adult credit diabetes \
            --model_name ${model_name} &> logs/011319_bias_var_${counter}.log &
    echo $((counter++))
done


# TODO: run calibration analysis

### Optional TODO: Resummarize the dataset to have importance attribute!
python -u summarize.py --identifier 0927 \
    --data_path results/091719_datasets.csv --d_name ${d_name} &> logs/0108_quantile_summarize_${counter}.log

python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name onlinenews \
        --model_name ${model_name} &> logs/120819_bias_var_onlinenews_${counter}.log &

# Make the outer bagging
d_name='adult breast churn credit heart mimicii mimiciii pneumonia compass support2cls2'
counter=400
for model_name in 'ebm-o100-i100-q' 'ebm-bf-o100-i100-q'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 1 \
        --d_name ${d_name} &> logs/0220_encoding_normal_datasets_${counter}.log && \
        python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} \
            &> logs/0220_encoding_normal_summarize_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/022020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

d_name='adult breast churn credit heart mimicii mimiciii pneumonia compass support2cls2'
counter=402
for model_name in 'xgb-d1-o100'; do
    (python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 1 \
        --d_name ${d_name} &> logs/0220_encoding_normal_datasets_${counter}.log && \
        python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name} \
            &> logs/0220_encoding_normal_summarize_${counter}.log && \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5 \
        &> logs/022020_compass_fimp_add_logloss_${counter}.log \
    ) &
    echo $((counter++))
done

#### Bias variance tradeoff
# TORUN
counter=801
model_name='xgb-d1-o100'
d_name='adult breast churn credit heart mimicii mimiciii pneumonia compass support2cls2'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &> logs/022220_bias_var_xgbo100_${counter}.log &

# int1 => 
counter=800
model_name='ebm-bf-o100-i100-q ebm-o100-i100-q'
d_name='adult breast churn credit heart mimicii mimiciii pneumonia compass support2cls2'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &> logs/022220_bias_var_ebmo100_${counter}.log &


## changing the plan => only runs the pneumonia
counter=900
model_name='ebm-bf-o100-i100-q ebm-o100-i100-q xgb-d1-o100'
d_name='pneumonia'
python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &> logs/022220_bias_var_ebmo100_${counter}.log &


##### SS datasets
# int1 => all the new SS dataset for xgb
counter=600
model_name='xgb-d1-o100 ebm-bf-o100-i100-q ebm-o100-i100-q ilr-q mlr-q lr flam spline-v2'
for d_name in \
'ss_churn_b1_r1377_xgb-d1-o100 ss_heart_b1_r1377_xgb-d1-o100' \
'ss_pneumonia_b1_r1377_xgb-d1-o100  ss_mimicii_b1_r1377_xgb-d1-o100' \
'ss_breast_b1_r1377_xgb-d1-o100 ss_mimiciii_b1_r1377_xgb-d1-o100'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0222_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0222_ss_summarize_${counter}.log) &
    echo $((counter++))
done

# int1 => all the dataset for ebm-o100-i100-q
counter=650
model_name='xgb-d1-o100 ebm-bf-o100-i100-q ebm-o100-i100-q ilr-q mlr-q lr flam spline-v2'
for d_name in \
'ss_churn_b1_r1377_ebm-o100-i100-q ss_heart_b1_r1377_ebm-o100-i100-q' \
'ss_pneumonia_b1_r1377_ebm-o100-i100-q  ss_mimicii_b1_r1377_ebm-o100-i100-q' \
'ss_breast_b1_r1377_ebm-o100-i100-q ss_mimiciii_b1_r1377_ebm-o100-i100-q'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0222_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0222_ss_summarize_${counter}.log) &
    echo $((counter++))
done

# int2 running: All the old dataset, run the ebm-o100-i100-q ebm-bf-o100-i100-q xgb-d1-o100
counter=700
d_name='ss_churn_b1_r1377_flam ss_heart_b1_r1377_flam ss_pneumonia_b1_r1377_flam  ss_mimicii_b1_r1377_flam 
ss_breast_b1_r1377_flam ss_mimiciii_b1_r1377_flam ss_churn_b1_r1377_spline-v2 ss_heart_b1_r1377_spline-v2
ss_pneumonia_b1_r1377_spline-v2  ss_mimicii_b1_r1377_spline-v2 ss_breast_b1_r1377_spline-v2 ss_mimiciii_b1_r1377_spline-v2
ss_churn_b1_r1377_lr ss_heart_b1_r1377_lr ss_pneumonia_b1_r1377_lr  ss_mimicii_b1_r1377_lr 
ss_breast_b1_r1377_lr ss_mimiciii_b1_r1377_lr'
for model_name in 'ebm-o100-i100-q ebm-bf-o100-i100-q xgb-d1-o100'; do
    (python -u main.py --identifier 1108_ss --n_splits 5 --test_size 0.15 --model_name ${model_name} \
        --d_name ${d_name} &> logs/0222_ss_${counter}.log && \
     python -u summarize.py --identifier 1112 \
        --data_path results/1108_ss.csv --d_name ${d_name} &> logs/0222_ss_summarize_${counter}.log) &
    echo $((counter++))
done


# Run in the srun
counter=900
model_name='xgb-d1-o100'
d_name='adult breast churn credit heart mimicii mimiciii compass support2cls2'
./srun.sh -o logs/022220_bias_var_xgbo100_${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &


# model_name='ebm-bf-o100-i100-q ebm-o100-i100-q'
counter=950
d_name='adult breast churn heart mimicii mimiciii compass support2cls2 credit'
for model_name in 'ebm-bf-o100-i100-q' 'ebm-o100-i100-q'; do
    ./srun.sh -o logs/022220_bias_var_ebmo100_${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done
counter=950
d_name='credit'
for model_name in 'ebm-bf-o100-i100-q'; do
    srun --mem=40G --gres=gpu:0 -c 80 -l -p cpu -o logs/022220_bias_var_ebmo100_${counter}.log python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done


# Run XGB-L2 (xgb-d1-o100-cols0.00001)
# d_name='adult breast churn credit heart mimicii mimiciii pneumonia compass support2cls2'
# counter=999
counter=1234
for d_name in 'adult breast churn heart mimicii mimiciii pneumonia compass support2cls2' 'credit'; do
for model_name in 'xgb-d1-o100-cols0.00001'; do
    (
        (./srun.sh -c 20 -o logs/0220_encoding_normal_datasets_${counter}.log \
        python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 1 \
        --d_name ${d_name}) && \
        (./srun.sh -c 5 -o logs/0220_encoding_normal_summarize_${counter}.log \
        python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name}) && \
        (./srun.sh -c 20 -o logs/022020_compass_fimp_add_logloss_${counter}.log \
        python feature_importances.py --data_path results/091719_datasets.csv \
        --identifier 0210_add --exp_mode AddExp --metric logloss \
        --model_name ${model_name} --d_name ${d_name} --end_splits 5)
    ) &
    echo $((counter++))
done
done

# No access to pneumonia now...
counter=1001
model_name='xgb-d1-o100-cols0.00001'
for d_name in 'adult breast churn heart mimicii compass support2cls2' 'mimiciii credit'; do
    srun --mem=40G --gres=gpu:0 -c 50 -l -p cpu -o logs/022220_bias_var_ebmo100_${counter}.log \
    python -u main.py --identifier 112819_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 8 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done


counter=1
model_name='xgb-d1-o100 ebm-o100-i100-q spline-v2 flam lr'
for d_name in \
'ss_heart_b1_r1377_ebm-o100-i100-q ss_heart_b1_r1377_xgb-d1-o100 ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2' \
'ss_breast_b1_r1377_ebm-o100-i100-q ss_breast_b1_r1377_xgb-d1-o100 ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2' \
; do
    srun --mem=30G --gres=gpu:0 -c 50 -l -p cpu -o logs/200602_ss_bias_var_${counter}.log \
    python -u main.py --identifier 060220_ss_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 5 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done

# Forget to run baseline: run a quick baseline here
counter=11
model_name='ebm-bf-o100-i100-q ilr-q mlr-q'
for d_name in \
'ss_heart_b1_r1377_ebm-o100-i100-q ss_heart_b1_r1377_xgb-d1-o100 ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2
ss_breast_b1_r1377_ebm-o100-i100-q ss_breast_b1_r1377_xgb-d1-o100 ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2' \
; do
    srun --mem=30G --gres=gpu:0 -c 50 -l -p cpu -o logs/200602_ss_bias_var_${counter}.log \
    python -u main.py --identifier 060220_ss_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 5 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done

counter=21
model_name='xgb-d1-o100 ebm-o100-i100-q flam lr ebm-bf-o100-i100-q ilr-q mlr-q'
for d_name in \
'ss_pneumonia_b1_r1377_ebm-o100-i100-q ss_pneumonia_b1_r1377_xgb-d1-o100 ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam ss_pneumonia_b1_r1377_spline-v2' \
'ss_churn_b1_r1377_ebm-o100-i100-q ss_churn_b1_r1377_xgb-d1-o100 ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam ss_churn_b1_r1377_spline-v2' \
'ss_mimicii_b1_r1377_ebm-o100-i100-q ss_mimicii_b1_r1377_xgb-d1-o100 ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2' \
'ss_mimiciii_b1_r1377_ebm-o100-i100-q ss_mimiciii_b1_r1377_xgb-d1-o100 ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam ss_mimiciii_b1_r1377_spline-v2' \
; do
    srun --mem=40G --gres=gpu:0 -c 50 -l -p cpu -o logs/200602_ss_bias_var_${counter}.log \
    python -u main.py --identifier 060220_ss_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 5 \
        --d_name ${d_name} \
        --model_name ${model_name} &
    echo $((counter++))
done

counter=31
model_name='spline-v2'
d_name=\
'ss_pneumonia_b1_r1377_ebm-o100-i100-q ss_pneumonia_b1_r1377_xgb-d1-o100 ss_pneumonia_b1_r1377_lr ss_pneumonia_b1_r1377_flam ss_pneumonia_b1_r1377_spline-v2
ss_churn_b1_r1377_ebm-o100-i100-q ss_churn_b1_r1377_xgb-d1-o100 ss_churn_b1_r1377_lr ss_churn_b1_r1377_flam ss_churn_b1_r1377_spline-v2
ss_mimicii_b1_r1377_ebm-o100-i100-q ss_mimicii_b1_r1377_xgb-d1-o100 ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2
ss_mimiciii_b1_r1377_ebm-o100-i100-q ss_mimiciii_b1_r1377_xgb-d1-o100 ss_mimiciii_b1_r1377_lr ss_mimiciii_b1_r1377_flam ss_mimiciii_b1_r1377_spline-v2'
srun --mem=40G --gres=gpu:0 -c 50 -l -p cpu -o logs/200602_ss_bias_var_${counter}.log \
python -u main.py --identifier 060220_ss_bias_var --exp_mode bias_var --n_splits 5 --n_subsamples 5 \
    --d_name ${d_name} \
    --model_name ${model_name} &
echo $((counter++))


ss_mimicii_b1_r1377_ebm-o50-i50 ss_mimicii_b1_r1377_ebm-bf-o50 ss_mimicii_b1_r1377_xgb-d1-o20-cv ss_mimicii_b1_r1377_lr ss_mimicii_b1_r1377_flam ss_mimicii_b1_r1377_spline-v2
ss_heart_b1_r1377_ebm-o50-i50 ss_heart_b1_r1377_ebm-bf-o50 ss_heart_b1_r1377_xgb-d1-o20-cv ss_heart_b1_r1377_lr ss_heart_b1_r1377_flam ss_heart_b1_r1377_spline-v2" \
"ss_breast_b1_r1377_ebm-o50-i50 ss_breast_b1_r1377_ebm-bf-o50 ss_breast_b1_r1377_xgb-d1-o20-cv ss_breast_b1_r1377_lr ss_breast_b1_r1377_flam ss_breast_b1_r1377_spline-v2
ss_adult_b1_r1377_ebm-o50-i50 ss_adult_b1_r1377_ebm-bf-o50 ss_adult_b1_r1377_xgb-d1-o20-cv ss_adult_b1_r1377_lr ss_adult_b1_r1377_flam ss_adult_b1_r1377_spline-v2

# bias / varinace analysis on the semi-synthetic datasets; see how it correlates well with the recovery error

# Resummarize the graph in the suppoer2cls for new models ebm and xgb-d1-o100
counter=11
d_name='support2cls'
for model_name in 'xgb-d1-o100' 'ebm-o100-i100-q' 'ebm-bf-o100-i100-q'; do
    (srun --mem=30G --gres=gpu:0 -c 30 -l -p cpu -o logs/200604_normal_datasets_${counter}.log \
        python -u main.py --identifier 091719_datasets --n_splits 5 --test_size 0.15 \
        --model_name ${model_name} --overwrite 1 \
        --d_name ${d_name}) && \
    (srun --mem=16G --gres=gpu:0 -c 5 -l -p cpu -o logs/200604_normal_summarize_${counter}.log \
    python -u summarize.py --identifier 0927 --overwrite 1 \
        --data_path results/091719_datasets.csv --d_name ${d_name} --model_name ${model_name}) &
    echo $((counter++))
done

