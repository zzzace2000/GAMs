conda create --name gam python=3.7 -y

conda activate gam

conda install pandas numpy scikit-learn==0.21.2 seaborn -y
conda install -c r r rpy2 r-mgcv -y
conda install -c conda-forge xgboost -y
R -e 'install.packages(c("flam"), repos="https://cloud.r-project.org/")'

pip install pygam

git clone https://github.com/zzzace2000/interpret my_interpret
cd my_interpret/
bash ./build.sh
cd ..


