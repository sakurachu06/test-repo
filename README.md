# sp-concreteML-WDTCRecurrenceClassifier

## System Description
The client-server system presented here is a fully homomorphic encryption-based XGBoost model for binary classification of WDTC recurrence that uses patient health data as its input. It allows clients to preprocess and encrypt their health data then send them to the server for FHE inference. The system was developed on Windows Subsystem for Linux 2 (WSL2). The client-side consists of a graphical user interface (GUI) mainly built with Tkinter, CustomTkinter. The server-side is a web application built with Django web framework. Both sides of the system are also built using Concrete ML library for the implementation of main functionalities: key generation, encryption, decryption, and FHE inference.

## Installation and Set Up
Users may set up the system by:
Manual Set Up
   - Please refer to the [Development Manual](https://github.com/sakurachu06/test-repo/blob/main/manual/DEVELOPMENT.md) for instructions on setting up the system from scratch.

Once done with the system set-up, users can refer to the [User Manual](https://github.com/sakurachu06/test-repo/blob/main/manual/HELP.md) for instructions on running the system. 
