from django.conf import settings
import datetime, os

def create_log(message):
    current_date = datetime.datetime.now()
    log_path = "{}/storages/logs".format(settings.BASE_DIR)
    log_filename = f"logs-{current_date.year}-{current_date.month:02d}-{current_date.day:02d}.txt"
    
    with open("{}/{}".format(log_path, log_filename), "a+") as log_file:
        log_file.write(f"{current_date} - {message}\n")

def remove_image(filepath):
    if os.path.isfile(filepath) and os.path.exists(filepath):
        os.remove(filepath)