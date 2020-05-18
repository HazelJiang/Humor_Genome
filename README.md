This Repository is Deprecated, refer to cmdid-webapp-backend and cmdid-webapp-frontend repo


Find the Comedian
===================================

Georgia Tech Institute of Technology
VIP-Humor Genome Project
ML - Categorization Team

### Python Version
Version 3

## Resources

## Setup Guide

Note: This instruction is for macos only

1. open terminal, run 'git --version'. if you already have git, terminal should output "git version $your_git_version". if you don't have git, it should prompt you to install it, or you can install git by downloading the installer on this website "https://git-scm.com/download/mac"

2. cd to a directory that you want to have the project in.

3. run "git clone https://github.com/cchu70/ComedianIdentifier.git"

4. open the above-mentioned directory in your finder.

5. you should see a new folder named "ComedianIdentifier"

6. don't go into the 'ComedianIdentifier' folder just yet, unzip the zip file downloaded from channel to current folder, go inside the folder created by unzipping the zip file, copy the two files inside the folder(download.py and setup.sh), go back to the previous folder that has the "ComedianIdentifier" folder. 

7. make sure that you have install the python3.7 by run "python3.7" in the terminal. this should put you into python's interactive shell. you can type "quit()" to close it. if command not found, go install python3.7. 

8. open your terminal, cd to the folder containing the ComedianIdentifier folder, download.py, and setup.sh.

9. run "source setup.sh" 

10. after a while, you should see nltk downloader gui window, select the first row to download all data and click download. if a warning message show up "ssl fail", close the window, and go to step a1.

11. after the download is complete, you should see "(env)lawn-xxx-xxx-xx-xxx" as the beginning of your command line input. this means you are in the virtual environment we just created. you should also see a new folder in the directory named "env". To deactivate the environment, type "deactivate". To re-activate the environment, make sure you are in the directory containing the "env" folder, and run "source env/bin/activate".

12. done.

a1. run "/Applications/Python\ 3.7/Install\ Certificates.command".

a2. go back to step 8
