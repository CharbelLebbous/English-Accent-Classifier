# English-Accent-Classifier

USE VSCODE TO AVOID UNCOMPATIBILY WHEN RUNNING.

To test our agent: You can also watch the Demo.mp4

The trained model is in the model folder.

Step1:
    Install the needed libraries for the main-pipline.py file in a venv
    
    Core Dependencies:
    numpy
    pandas
    joblib
    requests
    librosa
    matplotlib
    seaborn
    scikit-learn
    moviepy
    yt-dlp
    whisper
    transformers
    torch
    tk
    ffmpeg       

Step2:

    Then go into the main-pipline.py file and change the path of the model and the scaler .pkl files here:
    line 130-131:
        # Load model
            scaler = joblib.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\scaler.pkl")
            model = joblib.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\random_forest_accent_classifier_tune.pkl")

Step3:            
    Then run main-pipline.py

    command: python main-pipline.py 
    To visulize the simple UI, where you will insert your url link of the video and wait for the english accent prediction and summary.


To preprocess the dataset (optional)

Step1:
    Dataset name: Common Voice Delta Segment 10.0
    Dataset link: https://commonvoice.mozilla.org/en/datasets

    Download the dataset and unzip it

Step2:
    Then go into the datapreproccessing.py file and change these paths files here to be compatibale with your directories (corresponding file paths):

    line 12-13
        # === Paths ===
        metadata_file = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\cv-corpus-10.0-delta-2022-07-04\\en\\validated.tsv"
        audio_dir = "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\cv-corpus-10.0-delta-2022-07-04\\en\\clips"

    line 138
    # Save scaler
        joblib.dump(scaler, 'D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\scaler.pkl')

Step3:            
    Then run datapreproccessing.py

    command: python datapreproccessing.py 
    And enjoy the terminal information.

To train the model (optional) since i provided the trained model:

Step1:
    Go into the train.py file and change these paths files here to be compatibale with your directories (corresponding file paths):

        line 12-13
            # === Load normalized features and labels ===
            X = np.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\features_balanced_normalized.npy")
            y = np.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\labels_balanced.npy")

        line 71
            # === Save final trained model ===
            joblib.dump(rf, "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\random_forest_accent_classifier_tuned.pkl")

Step2:            
    Then run train.py

    command: python train.py 
    And enjoy the terminal information and charts to monitor the training and evaluating the model performance.
