Initial Setup
=============

After installation of **MeerSOLAR**, before running the pipeline, some initial setup is needed. These include downloading some required metadata for the pipeline and setup of remote logger.


Download MeerSOLAR metadata
---------------------------
1. To download and save the required MeerSOLAR metadata in appropriate directory, run from command line:

.. code-block :: bash
    
    init_meersolar_data --init
    
2. If data files are present, but needs to updated, run:

.. code-block :: bash

    init_meersolar_data --init --update
    
Setup e-mail ids
----------------
To receive remote logger Job ID and password, use can setup their e-mail id(s) in MeerSOLAR. 

.. code-block :: bash

    init_meersolar_data --init --emails <youremail1@email1.id1>,<youremail2@email2.id2> 
    
If you setup a remote logger as described below, you will receive a Job ID and auto-generated six-character password to access logs of a particular pipeline run from the remote logger. Without this password, one can not access logs of that particular pipeline run. This added security as well as privacy when multiple user uses the same remote logger link, for example, an institute based remote logger link.   
    
Setup remote logger link
-------------------------
If remote logger is intended to be used, setup the remote link in MeerSOLAR metadata.

.. code-block :: bash
    
    init_meersolar_data --init --remotelink https://<remote-logger-name>.onrender.com
    
Before doing this, create your own remote logger on free-tier cloud platform, https://render.com. One can use, same **remotelink** in multiple machines and users. However, free-tier link has some limitations on bandwidth. If you want to use **remotelink** for your institution, we suggest to purchase suitable paid version or setup seperate **remotelink** for different users.

Update remote logger link and/or e-mail ids
-------------------------------------------
If user wants to update the already provided remote logger link or e-mail id(s), simply run the above commands with new values. MeerSOLAR will automatically update the database with these new values.

Tutorial to setup remote lor link
---------------------------------
1. Go to https://dashboard.render.com/. It will take you to the login page. If you do not have an ccount on https://render.com, create an account and login.

2. After login, you will land up in the following page. Click on **Add new** and then **Web Service**.

.. image :: _static/ss1.png

3. Then the following page will open. Select **Exisiting image** table and put **docker.io/devojyoti96/meerlogger:latest** in the **Image URL** box. If the image link is correct, a blue tock will appear the right corner of the box as hown in the image below. Then click on black **Connect-->** button at the bottom of the page.

.. image :: _static/ss2.png

4. Next, you will land up in the following page. In the name box, type your custom remote logger name, **<remote-logger-name>**. If the name is in use by anyone else and not available, a red text will appear below the box showing **Name is already in use**. Modify the name to have a unique available name. Once done, scroll down to the middle of the page.

.. image :: _static/ss3.png

5. In the middle of the page, select the user-plan. For individual user, you can choose **Free** as shown in the image below. If you want any paid version, choose the appropriate one. 

.. image :: _static/ss5.png

6. Then go down to the bottom of the page. Click of **Deploy** button.

.. image :: _static/ss4.png

7. In the next page, you will see web-service is being started. Once you see, **==> Your service is live 🎉** as shown in the image below, you remote logger is ready to use. You remote logger link is also shown just above the black window, and it will be **https://<remote-logger-name>.onrender.com**.

.. image :: _static/ss6.png

8. Now use this link to setup as remote logger link as described above. This link is persistent and can be used in multiple machines. You MeerSOLAR job logs will appear in **https://<remote-logger-name>.onrender.com**. How to access 

























    
