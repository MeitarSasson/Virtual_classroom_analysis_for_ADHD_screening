import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from glob import glob
import customtkinter
from couchdb import Server
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import couchdb
import json
# from Gui.HomePage import MainApp
from backend import Model
from main import PaddedFrame

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


def extract_values_from_filename(filename):
    directory = os.path.dirname(filename)
    class_name = os.path.basename(directory)

    # Check if the class name is 'ADHD' or 'NOADHD'
    if class_name == 'ADHD':
        return 1
    elif class_name == 'NOADHD':
        return 0
    else:
        raise ValueError(f"Invalid directory name: {class_name}. Expected 'ADHD' or 'NOADHD'.")


class ImportFiles(PaddedFrame):
    def __init__(self, master, parent_frame=None, model=None, callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.parent_frame = parent_frame
        self.model = model if model else Model()  # Use the provided model or create a new one
        self.callback = callback
        self.grid(sticky='nsew', padx=20, pady=20)
        self.master.grid_rowconfigure(0, weight=1, pad=20)
        self.master.grid_columnconfigure(0, weight=1, pad=20)
        self.db = couchdb.Server('http://admin:1234@localhost:5984/')['test_results']
        self.next_id = self.get_next_id()
        self.label = ttk.Label(self, font=("Helvetica", 20), text="Import Files", justify=tk.CENTER)
        self.label.grid(row=0, column=0)  # use grid instead of pack

        self.button_class0 = ttk.Button(self, text="Import Class 0 Files",
                                        command=lambda: self.import_files(0))
        self.button_class0.grid(row=1, column=0)  # use grid instead of pack

        self.button_class1 = ttk.Button(self, text="Import Class 1 Files",
                                        command=lambda: self.import_files(1))
        self.button_class1.grid(row=1, column=1)  # use grid instead of pack

        self.continue_button = ttk.Button(self, text="Continue", command=self.on_import_complete)
        self.continue_button.grid(row=2, column=0, sticky='')  # place button in the middle row

        self.back_button = ttk.Button(self, text="Back",
                                      command=self.go_back)
        self.back_button.grid(row=2, column=1)

        # Connect to CouchDB
        self.couch = couchdb.Server('http://admin:1234@localhost:5984/')  # replace with your server URL if not local

        # Select or create the 'test_results' database
        self.db_name = 'test_results'
        if self.db_name in self.couch:
            self.db = self.couch[self.db_name]
        else:
            self.db = self.couch.create(self.db_name)

    def get_next_id(self):
        # Get the highest existing ID in the database
        result = self.db.view('_all_docs', descending=True, limit=1)
        rows = list(result)
        if rows:
            highest_id = int(rows[0].id)
        else:
            highest_id = -1  # If the database is empty, start from 0
        return highest_id + 1

    def go_back(self):
        if self.callback:
            self.callback()

    def import_files(self, class_num):
        files = filedialog.askopenfilenames(filetypes=[("JSON files", "*.json")])
        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
            data['ADHD'] = class_num
            self.db[str(self.next_id)] = data
            self.next_id += 1

    def on_import_complete(self):
        messagebox.showinfo("Import Complete", "Data added successfully!")
        if self.callback:
            self.callback()


class TrainWindow(PaddedFrame):
    def __init__(self, master, parent_frame=None, model=None, callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.parent_frame = parent_frame
        self.model = model if model else Model()  # Use the provided model or create a new one
        self.callback = callback
        self.grid(sticky='nsew', padx=20, pady=20)
        self.master.grid_rowconfigure(0, weight=1, pad=20)  # use grid instead of pack
        self.master.grid_columnconfigure(0, weight=1, pad=20)

        self.label_5 = ttk.Label(self, font=("Helvetica", 20), text="Training Page", justify=tk.CENTER)
        self.label_5.grid(row=0, column=0)  # use grid instead of pack

        self.library_class0 = None
        self.library_class1 = None
        self.check_vars_class0 = []
        self.check_vars_class1 = []

        self.button_select_dir_class0 = ttk.Button(self, text="Select Directory NOADHD",
                                                   command=lambda: self.select_directory_and_display_files(
                                                       0))
        self.button_select_dir_class0.grid(row=1, column=0)  # use grid instead of pack

        self.button_select_dir_class1 = ttk.Button(self, text="Select Directory ADHD",
                                                   command=lambda: self.select_directory_and_display_files(
                                                       1))
        self.button_select_dir_class1.grid(row=1, column=1)  # use grid instead of pack

        self.start_training_button = ttk.Button(self, text="Start Training", command=self.start_training)
        self.start_training_button.grid(row=2, column=0, sticky='')  # place button in the middle row

        self.back_button = ttk.Button(self, text="Back",
                                      command=self.go_back)
        self.back_button.grid(row=2, column=1)

    def fetch_files_from_db(self, class_num):
        server = Server('http://admin:1234@localhost:5984/')
        db = server['test_results']
        files = [db[doc_id] for doc_id in db if db[doc_id]['ADHD'] == class_num]
        return files

    def go_back(self):
        if self.callback:
            self.callback()

    def select_directory_and_display_files(self, class_num):
        files = self.fetch_files_from_db(class_num)
        self.display_files(files, class_num)

    def get_json_files(self, library):
        json_files = [os.path.join(library, file) for file in os.listdir(library) if file.endswith(".json")]
        return json_files

    def display_files(self, files, class_num):
        offset = 5  # Offset to start displaying filenames a row after the start training button
        if class_num == 0:
            self.check_vars_class0 = []
            for index, file in enumerate(files):
                var = tk.BooleanVar()
                check = tk.Checkbutton(self, text=file['_id'], variable=var)
                check.grid(row=index + offset, column=class_num, sticky='w')
                self.check_vars_class0.append((var, file))
        elif class_num == 1:
            self.check_vars_class1 = []
            for index, file in enumerate(files):
                var = tk.BooleanVar()
                check = tk.Checkbutton(self, text=file['_id'], variable=var)
                check.grid(row=index + offset, column=class_num, sticky='w')
                self.check_vars_class1.append((var, file))

    def start_training(self):
        selected_files_class0 = [file for var, file in self.check_vars_class0 if var.get()]
        selected_files_class1 = [file for var, file in self.check_vars_class1 if var.get()]

        # Ensure both classes have been selected
        if not selected_files_class0:
            messagebox.showerror("Error", "No files selected for Class 0.")
            return

        if not selected_files_class1:
            messagebox.showerror("Error", "No files selected for Class 1.")
            return

        self.parent_frame.model = Model()
        self.parent_frame.model.add_files(selected_files_class0, selected_files_class1)
        self.parent_frame.model.train_model(selected_files_class0, selected_files_class1)

        messagebox.showinfo("Training", "Training complete!")

        if self.callback:
            self.callback()  # Call the callback function


class TestWindow(PaddedFrame):
    def __init__(self, master, parent_frame=None, model=None, callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.parent_frame = parent_frame
        self.model = model if model else Model()  # Use the provided model or create a new one
        self.callback = callback
        self.grid(sticky='nsew', padx=20, pady=20)
        self.master.grid_rowconfigure(0, weight=1, pad=20)
        self.master.grid_columnconfigure(0, weight=1, pad=20)

        self.figures = {}  # Store the figures here

        self.check_vars = []

        self.button_select_files = ttk.Button(self, text="Select Files",
                                              command=self.select_files_and_display)
        self.button_select_files.grid(row=0, column=0)

        # Start testing button
        self.start_testing_button = ttk.Button(self, text="Start Testing",
                                               command=self.start_testing)
        self.start_testing_button.grid(row=1, column=0)

        # Back button
        self.back_button = ttk.Button(self, text="Back",
                                      command=self.go_back)
        self.back_button.grid(row=2, column=0)

        self.selected_files = []

    def go_back(self):
        if self.callback:
            self.callback()

    def fetch_files_from_db(self):
        server = Server('http://admin:1234@localhost:5984/')
        db = server['test_results']
        files = [doc for doc in db]
        return files

    def start_testing(self):
        selected_files = [file for var, file in self.check_vars if var.get()]
        if not selected_files:
            messagebox.showerror("Error", "Please select at least one file.")
            return

        self.model.add_test_data(selected_files)
        self.figures["Prediction Plot"], self.figures["Confusion Matrix"] = self.model.test_model()
        print("y_pred probabilities:\n", self.model.y_pred_proba)
        # Create a string with the probabilities for ADHD
        # probabilities_string = ""
        # for i, prob in enumerate(self.model.y_pred_proba):
        #     if i < len(selected_files):
        #         filename = selected_files[i]
        #         probabilities_string += f"Session ID {os.path.splitext(os.path.basename(filename))[0]} returned a {prob * 100:.2f}% chance of having ADHD\n"
        #     else:
        #         probabilities_string += f"Session ID {i + 1} has returned a {prob * 100:.2f}% chance of having ADHD\n"

        # Display the probabilities in a tkinter message box
        messagebox.showinfo("Testing", "Testing complete!")

        # Disable the start testing button
        self.start_testing_button.configure(state='disabled')

        # Check if confusion matrix is None and set the dropdown menu options accordingly
        if self.figures["Confusion Matrix"] is None:
            dropdown_options = ["Prediction Plot"]
        else:
            dropdown_options = ["Prediction Plot", "Confusion Matrix"]

        # Create the dropdown menu
        self.plot_options = ttk.Combobox(self, values=dropdown_options)
        self.plot_options.bind("<<ComboboxSelected>>", self.display_selected_plot)
        self.plot_options.grid(row=2, column=1)

    def select_files_and_display(self):
        files = self.fetch_files_from_db()
        self.display_files(files)

    def select_directory_and_display_files(self, class_num):
        library = filedialog.askdirectory()
        files = self.get_json_files(library)
        self.display_files(files)

    def get_json_files(self, library):
        json_files = [os.path.join(library, file) for file in os.listdir(library) if file.endswith(".json")]
        return json_files

    def display_files(self, files):
        offset = 5  # Offset to start displaying filenames a row after the start testing button
        self.check_vars = []
        for index, file in enumerate(files):
            var = tk.BooleanVar()
            check = tk.Checkbutton(self, text=os.path.basename(file), variable=var)
            check.grid(row=index + offset, column=0, sticky='w')
            self.check_vars.append((var, file))

    def display_selected_plot(self, event):
        # Retrieve the selected option
        selected_plot = self.plot_options.get()

        # Clear the frame
        for widget in self.winfo_children():
            if widget != self.plot_options and widget != self.back_button:
                # Don't destroy the Combobox and the back button
                widget.destroy()

        # Create the canvas and display the selected figure
        canvas = FigureCanvasTkAgg(self.figures[selected_plot], master=self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)


class developerPage(PaddedFrame):
    def __init__(self, master, parent_frame=None, model=None, callback=None, **kwargs):
        kwargs.pop('parent', None)  # Remove the 'parent' key from kwargs
        super().__init__(master, **kwargs)
        self.master = master
        self.parent_frame = parent_frame
        self.model = Model()  # Use the provided model or create a new one
        self.callback = callback
        self.grid(sticky='nsew', padx=20, pady=20)
        self.master.grid_rowconfigure(0, weight=1, pad=20)
        self.master.grid_columnconfigure(0, weight=1, pad=20)

        label_4 = ttk.Label(self, font=("Helvetica", 20), text="Developer Page", justify=tk.LEFT)
        label_4.grid(row=0, column=0)  # use grid instead of pack

        self.combobox_4 = ttk.Combobox(self, values=["Import data", "Train model", "Test model"])
        self.combobox_4.grid(row=1, column=0, columnspan=2)  # use grid instead of pack
        self.combobox_4.set("Choose Action")

        button_6 = ttk.Button(self, text="Continue", command=self.choose_action)
        button_6.grid(row=2, column=0, columnspan=2)  # use grid instead of pack

        button_7 = ttk.Button(self, text="Back", command=self.callback)
        button_7.grid(row=3, column=0, columnspan=2)  # use grid instead of pack

    def choose_action(self):
        action = self.combobox_4.get()
        if not action or action == "Choose Action":
            messagebox.showerror("Action Required", "Please select an action")
            return

        if action == "Train model":
            self.master.switch_frame(TrainWindow, parent_frame=self, model=self.model,
                                     callback=self.on_operation_complete)
        elif action == "Test model":
            if not os.path.isfile('xgb_model.json'):  # Check if model file exists before testing
                messagebox.showinfo("Error", "Please train the model before testing!")
            else:
                self.master.switch_frame(TestWindow, parent_frame=self, model=self.model,
                                         callback=self.on_operation_complete)
        elif action == "Import data":
            self.master.switch_frame(ImportFiles, parent_frame=self, model=self.model,
                                     callback=self.on_operation_complete)

    def on_operation_complete(self):
        self.master.switch_frame(developerPage, model=self.model)

    def Log_Out(self):
        self.master.quit()

