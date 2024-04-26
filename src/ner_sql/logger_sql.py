# A data logger is created to capture bugs
class DataLogger:
    def __init__(self, file_name):
        self.file_name = file_name

    def create_logger(self):
        with open(self.file_name, 'w', newline='', encoding='utf-8') as file:
            file.write("Log file created successfully")

    def log_data(self, data):
        with open(self.file_name, 'a') as file:
            file.write(data + '\n')
        print("Data logged successfully.")


logger = DataLogger('data_log.txt')
