# pytorch_model_on_gcp

Google Cloud Function code for download the pytorch model from the Google Cloud Storage 

Google Function Settings
  Env: 1st Gen
  Runtime: python3.7
  Memory: 1 GB
  Trigger type: Cloud Storage
  Event type: On (finalizing/creating) file in the selected bucket


Telegram bot handler (php) --> jpg --> GCS bucket 
                                            |---trigger---> GCF main.py --> download pytorch model and jpg file from the bucket
