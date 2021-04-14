# Welcome to experiments on low rank approximation on speech 
This project is started and guided by Dr Vinayak Abrol.(1 April 2021) 

###Steps to be followed for setup 

1) Below are links of Research papers that we must read before proceeding further 
    1) http://publications.idiap.ch/downloads/reports/2019/Abrol_Idiap-RR-11-2019.pdf
        _this is sir's paper and new idea of low rank filtering which we need to implement._
    
    2) https://arxiv.org/pdf/2005.03191.pdf  _this is contextNet, over which we need to implement new ideas._
    
2) Links to Dataset  https://drive.google.com/drive/folders/1jf0wWOZ59YmOziRL37cQLlaPUys_DlqC?usp=sharing 
_This link has all the tar.gz dataset files, that would be needed. (7.5 gb)_
    1) dev-clean
    2) dev-other
    3) train-clean-100
    4) test-clean
    
3) We have access to quadro rtx 4000 VM. Download Anydesk software. For login credentials request sir.
    
4) There are two conda env on the machine. 
    1) **tf2**  (main env)
    2) **beta** (testing and experimenting purposes used by vaibhav singh)
    
5) Download the environment.yml and store in the current working directory, https://drive.google.com/drive/u/3/folders/19Ga67n5hRgBN6-Svpm1Uh6zQKFpye7YY
There is another file config.yml and create_transcripts.sh which will be used in the code further.  

6) For testing purposes create a new environment
    1) conda create -n {env_name} -f {path to environment.yml}
    2) conda activte {env_name}
    3) Setup of TensorflowASR 
        1) `git clone https://github.com/TensorSpeech/TensorFlowASR`
        2) do this git clone in the current directory. Lets say we are in /Desktop 
        3) Go through the README of the project. 
        4) Our current objective is to implement low rank conv module in the contextNet model
    
    4) Now go to /TensorFlowASR/scripts/ and create an empty folder Datasets. So we will now have /TensorFlowASR/scripts/Datasets
    5) Now create 4 empty folders dev-clean , dev-other , train-clean-100 , test-clean inside /TensorFlowASR/scripts/Datasets 
    6) Now run these 4 commands 
        Kindly check the directory and folder names. Purpose of this command is to have all the datasets in unzipped form in our project directory. 
        1) `tar -xvf /Desktop/Datasets/test-clean.tar.gz -C /Desktop/TensorFlowASR/scripts/Datasets/test_clean/`
        2) `tar -xvf /Desktop/Datasets/dev-clean.tar.gz -C /Desktop/TensorFlowASR/scripts/Datasets/dev_clean/`
        3) `tar -xvf /Desktop/Datasets/train-clean-100.tar.gz -C /Desktop/TensorFlowASR/scripts/Datasets/train_clean`
        4) `tar -xvf /Desktop/Datasets/dev-other.tar.gz -C /Desktop/TensorFlowASR/scripts/Datasets/dev_other`
        
7) Now our datasets are cool, so go to /Desktop/TensorFlowASR and run this command
    1) `pip3 install -r requirements.txt` 
    2) `sh create_transcripts.sh` (same link as environment.yml file )
    this will create transcripts. 

11) Now repalce the config.yml in Desktop/TensorFlowASR/examples/contextnet/ with the one donwloaded from link
We need to replace the datasets path here, according to your machine. Thats all

12) Just to be double-sure, this project gives a setup file too. Lets run it to be absolutely sure. 
    1) `python3  Desktop/TensorFlowASR/setup.py build`
    2) `python3  Desktop/TensorFlowASR/setup.py install`
    3) **WORD OF CAUTION**- whenever any change is made to the code and scripts are run, then you would need to run the above two commands first, because setuptools install
    the packages in env/lib/site-packages/python3.8/XXXX. So your change will not be reflected unless you build and install again. 

14)Since our requirements are already there, so it would skip and process would be finished with success.

13) Finally we will run our main script.
    1)`python3 /Desktop/TensorFlowASR/examples/contextnet/train_contextnet.py`
    
    2) since we need logging into files and in background to free terminal so better run this command
     `_nohup python3 /Desktop/TensorFlowASR/examples/contextnet/train_contextnet.py > logging.out &_`
     
    3) if you want error in a seperate file then 
         `_nohup python3 /Desktop/TensorFlowASR/examples/contextnet/train_contextnet.py > logging.out 2>&1 &_`

14) For testing run this script 
    1) `python3 /Desktop/TensorFlowASR/examples/contextnet/test_contextnet.py --saved /Desktop/TensorFlowASR/examples/contextnet/latest.h5`


## Contact

Dr Vinayak Abrol (Professor IIIT-D) _abrol@iiitd.ac.in_


Vaibhav Singh(Research Intern) _vaibhavsinghfcos@gmail.com_
