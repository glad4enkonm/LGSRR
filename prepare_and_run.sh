wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash Anaconda3-2025.06-1-Linux-x86_64.sh
git clone https://github.com/glad4enkonm/LGSRR
cd LGSRR/
conda create --name lgsrr python=3.9
conda activate lgsrr
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
bash examples/run_lgsrr_MIntRec2.sh