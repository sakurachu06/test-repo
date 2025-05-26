# User Manual

## Limitations
The system is currently limited to multi-class brain tumor classification and is trained on only five classes which are:
1. Normal
2. Ependymoma
3. Glioblastoma
4. Medulloblastoma
5. Pilocytic Astrocytoma

## Installation and System Set Up
Please refer to the [Development Manual](DEVELOPMENT.md) for instructions on setting up the system.

## Input File Structure
The client-server interaction will begin once the client submits its input data to the client GUI application. The input must be a CSV file consisting of brain cancer gene expression data with the following structure:
1. The input consists of a column for the sample's name or ID number. Value in this column may be a string (if name is used) or an integer (if ID is used).
2. The input has 54,675 columns representing genes. Value in these columns must be floats.

Note that the columns for genetic features must follow the same genes used in the training dataset available at https://www.kaggle.com/datasets/brunogrisci/brain-cancer-gene-expression-cumida. 

For demonstration and testing purposes, there are sample input files provided in the **testing-samples** folder found in the project folder.

## Running the Client-Server System
All commands in running the system are done in the terminal. 
- If using Windows:
  - In Windows File Explorer, open the folder of the cloned GitHub repository. Double click on the **src** folder to go to the source code directory. Open the WSL2 terminal by typing **bash** on the address bar then press **Enter**. Doing this will open the WSL2 terminal and automatically change the directory to your **src** folder. Do this twice to open two WSL2 terminals - one for the server and the other for the client.
- If using Ubuntu (Linux):
  - In Files, open the folder of the cloned GitHub repository. Right click on the **src** folder and select **Open in terminal**. This will open the terminal and automatically set the directory to the **src** folder. Do this twice to open two terminals - one for the server and the other for the client.

You can now follow these steps to run the client-server system.

1. Before using the client GUI application, it must be ensured that the server-side of the system is running. To do this, use the command:
    ``` {.bash language="bash"}
    python3 manage.py runserver
    ```
    This will run the server-side of the system. You may also load the server's homepage to make sure it is running by visiting [http:127.0.0.1:8000](http://127.0.0.1:8000/). The homepage contains a link to the system's GitHub repository.

2. In the second terminal you've opened, change the directory to the **client-gui** folder. This is going to be the terminal that you'll use in running the client-side of the system. Do this step by executing the command:
    ``` {.bash language="bash"}
    cd client-gui
    ```
3. Once you have successfully changed directories to **client-gui**, you can now run the client GUI application with the following command:
    ``` {.bash language="bash"}
    python3 client-app.py
    ```
    This will first download the required client files in the same directory then automatically launch the client GUI application.
    
4. After opening the client GUI, the client can now interact with the server for FHE inference. To submit an input data, click on the **Browse File** button of the application. You may use a file in the **testing-samples** folder when testing the application. Files provided here follows the input file structure required by the system.
    
5. Once you have selected an input file, click on the **Submit Data for FHE Classification** button of the application. This will automatically preprocess and encrypt your input file then send the encrypted file to the server for FHE inference. The server will then perform FHE inference on the encrypted data and send the result back to the client GUI application. Once the result has been received, the client app will automatically decrypt the prediction and display the final brain tumor classification result. You can see the progress of the process from preprocessing to displaying of the decrypted prediction in the output window of the client app. 
    
   The client GUI application also saves the decrypted FHE inference results into a CSV file stored in the **predictions** folder in the **client-gui** directory.
