# Getting Started with the Project

## Accessing the Dataset

To obtain access to the dataset, follow the steps below:

1. **Install DVC**  
   Install [DVC (Data Version Control)](https://dvc.org/doc/install) on your local machine by following the installation instructions provided on their website.

2. **Connect to the WMI UAM VPN**  
   Connect to the Faculty of Mathematics and Computer Science (WMI UAM) VPN following the instructions here:  
   [https://laboratoria.wmi.amu.edu.pl/uslugi/vpn/](https://laboratoria.wmi.amu.edu.pl/uslugi/vpn/)  
   > **Note:** You must be a student or employee of Adam Mickiewicz University (UAM) to gain access.

3. **Download the Dataset Using DVC**  
   Once connected to the VPN, open a terminal and run the following commands:
   ```bash
   cd Ophthalmic_Scans
   dvc pull
   ```

   This will download the dataset to your local machine.

**Tip:** Make sure you have the necessary permissions and that your VPN connection is active before running dvc pull.
