import os
import tkinter
from tkinter import ttk, filedialog, messagebox
import couchdb
import json

import pandas as pd
import tk
from couchdb import ResourceNotFound, Server

from Gui.ViewAnlyzeGraph import viewAnlyzeGraph
from Gui.ViewAnlyzeReport import viewAnlyzeReport
from main import PaddedFrame, BasePage


class CouchDBAPI:
    def __init__(self):
        # Download CouchDB
        # pip install couchdb
        # Connect to CouchDB server
        self.couch = Server('http://admin:1234@localhost:5984/')

        # Create or access a database named 'student_details'
        self.db = self.couch.create('student_details') if 'student_details' not in self.couch else self.couch[
            'student_details']

    def save_student_details(self, username, details):
        # Save the details to the database
        self.db[username] = details

    def get_student_details(self, username):
        # Retrieve the details from the database
        return self.db[username]

    def register_student(self, username, password, fullname, age, gender, email, adhd_test_details):
        doc = {
            '_id': username,
            'password': password,
            'fullname': fullname,
            'age': age,
            'gender': gender,
            'email': email,
            'adhd_test_details': adhd_test_details
        }
        try:
            self.db.save(doc)
            return True
        except couchdb.http.ResourceConflict:
            return False

    def login_student(self, username, password):
        # Retrieve the student registration details from the database
        try:
            details = self.db[username]
        except ResourceNotFound:
            return False

        # Check if the password matches
        if details["password"] == password:
            return True
        else:
            return False

    def get_student(self, username):
        # Retrieve the student from the database
        try:
            return self.db[username]
        except ResourceNotFound:
            return None


class StudentRegister(PaddedFrame):
    def __init__(self, parent, parent_frame, model=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.model = model
        self.callback = callback
        self.parent = parent

        # Create an instance of the CouchDB API
        self.backend_api = CouchDBAPI()
        self.label_title = ttk.Label(self, text="Register", font=("Helvetica", 25))
        self.label_title.grid(row=0, column=0)
        # Add your registration form fields here
        self.label_username = ttk.Label(self, text="Username:")
        self.label_username.grid(row=1, column=0)
        self.entry_username = ttk.Entry(self)
        self.entry_username.grid(row=1, column=1)

        self.label_password = ttk.Label(self, text="Password:")
        self.label_password.grid(row=2, column=0)
        self.entry_password = ttk.Entry(self, show="*")
        self.entry_password.grid(row=2, column=1)

        self.label_fullname = ttk.Label(self, text="Full Name:")
        self.label_fullname.grid(row=3, column=0)
        self.entry_fullname = ttk.Entry(self)
        self.entry_fullname.grid(row=3, column=1)

        self.label_age = ttk.Label(self, text="Age:")
        self.label_age.grid(row=4, column=0)
        self.entry_age = ttk.Entry(self)
        self.entry_age.grid(row=4, column=1)

        self.label_gender = ttk.Label(self, text="Gender:")
        self.label_gender.grid(row=5, column=0)
        self.entry_gender = ttk.Entry(self)
        self.entry_gender.grid(row=5, column=1)

        self.label_email = ttk.Label(self, text="Email:")
        self.label_email.grid(row=6, column=0)
        self.entry_email = ttk.Entry(self)
        self.entry_email.grid(row=6, column=1)

        self.button_register = ttk.Button(self, text="Register", command=self._register)
        self.button_register.grid(row=7, column=0, columnspan=2)

        self.back_button = ttk.Button(self, text="Back", command=self.callback)
        self.back_button.grid(row=8, column=0, columnspan=2)

    def _register(self):
        # Get the registration form data here
        username = self.entry_username.get()
        password = self.entry_password.get()
        fullname = self.entry_fullname.get()
        age = self.entry_age.get()
        gender = self.entry_gender.get()
        email = self.entry_email.get()

        # Check if all fields are filled
        if not all([username, password, fullname, age, gender, email]):
            messagebox.showerror("Registration failed", "All fields must be filled.")
            return

        # Open a file dialog to select the ADHD test details JSON file
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

        # Load the ADHD test details from the selected JSON file
        with open(filepath, "r") as file:
            adhd_test_details = json.load(file)
        self.file = adhd_test_details
        # Register the student using the API
        if self.backend_api.register_student(username, password, fullname, age, gender, email, adhd_test_details):
            # If registration is successful, switch to the StudentLogin frame
            self.parent.switch_frame(StudentLogin, parent_frame=self.parent_frame, model=self.model)
        else:
            # If registration failed, show an error message
            messagebox.showerror("Registration failed", "Unable to register. Please try again.")


class StudentLogin(PaddedFrame):
    def __init__(self, parent, parent_frame, model=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.model = model
        self.callback = callback
        self.parent = parent

        # add title label
        self.label_title = ttk.Label(self, text="Login Form", font=("Helvetica", 20))
        self.label_title.grid(row=0, column=0, sticky='w')
        # Create an instance of the CouchDB API
        self.backend_api = CouchDBAPI()

        # Add your login form fields here
        self.label_username = ttk.Label(self, text="Username:")
        self.label_username.grid(row=1, column=0, sticky='e')
        self.entry_username = ttk.Entry(self)
        self.entry_username.grid(row=2, column=0, columnspan=2)

        self.label_password = ttk.Label(self, text="Password:")
        self.label_password.grid(row=3, column=0, sticky='e')
        self.entry_password = ttk.Entry(self, show="*")
        self.entry_password.grid(row=4, column=0, columnspan=2)

        self.button_login = ttk.Button(self, text="Login", command=self.login)
        self.button_login.grid(row=5, column=0, columnspan=2)

        self.button_back = ttk.Button(self, text="Back", command=self.callback)
        self.button_back.grid(row=6, column=0, columnspan=2)

    def login(self):
        # Get the login form data here
        username = self.entry_username.get()
        password = self.entry_password.get()

        # Check if username or password is empty
        if not username or not password:
            messagebox.showerror("Login failed", "Username and password must be entered")
            return

        # Login the student using the API
        if self.backend_api.login_student(username, password):
            # Login successful
            # Get the student details
            student = self.backend_api.get_student(username)

            # Switch to the student details page
            self.parent.switch_frame(StudentPage, parent_frame=self.parent_frame, model=self.model,
                                     adhd_test_details=student)
        else:
            # Login failed
            # Show an error message
            messagebox.showerror("Login failed", "Invalid username or password")


class studentDetails(PaddedFrame):
    def __init__(self, parent, parent_frame, model=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.model = model
        self.callback = callback
        self.parent = parent

        # Create an instance of the CouchDB API
        self.backend_api = CouchDBAPI()

        # Create an empty dictionary to store the student details
        self.student_details = {}

        self.label_2 = ttk.Label(master=self, font=("Helvetica", 25), text="Student Screen", justify=tkinter.LEFT)
        self.label_2.grid(row=0, column=0, pady=10, padx=10)

        self.button_login = ttk.Button(self, text="Login", command=self.go_to_login)
        self.button_login.grid(row=1, column=0)

        self.button_register = ttk.Button(self, text="Register", command=self.go_to_register)

        self.button_register.grid(row=2, column=0)
        self.button_3 = ttk.Button(master=self, text="Back", command=self.callback)
        self.button_3.grid(row=3, column=0)

    def go_to_login(self):
        self.master.switch_frame(StudentLogin, parent_frame=self.parent, callback=self.parent.switch_frame)

    def go_to_register(self):
        self.master.switch_frame(StudentRegister, parent_frame=self.parent, callback=self.parent.switch_frame)

    def login(self):
        self.master.switch_frame(StudentLogin, parent=self.parent, callback=self.parent.switch_frame)

    def _register(self):
        self.master.switch_frame(StudentRegister, parent=self.parent, callback=self.parent.switch_frame)


class StudentPage(PaddedFrame):
    def __init__(self, parent, parent_frame, adhd_test_details=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.callback = callback
        self.parent = parent
        self.adhd_test_details = adhd_test_details  # Use the adhd_test_details parameter

        # Create an instance of the CouchDB API
        self.backend_api = CouchDBAPI()
        self.label_1 = ttk.Label(master=self, font=("Helvetica", 30), text="Student Details", justify=tkinter.LEFT)
        self.label_1.grid(row=0, column=0, pady=10, padx=10)

        # Create a combobox for session type selection
        self.session_type = tkinter.StringVar()
        self.session_type_combobox = ttk.Combobox(self, textvariable=self.session_type)
        self.session_type_combobox['values'] = ('Without Disturbances', 'With Disturbances')
        self.session_type_combobox.grid(row=1, column=0, columnspan=2, pady=10, padx=10)
        self.session_type_combobox.bind("<<ComboboxSelected>>", self.display_dataframe_data)

        # Create a frame for the student details
        self.details_frame = ttk.Frame(self)
        self.details_frame.grid(row=2, column=0, columnspan=2, sticky='nsw')

        # Create a frame for the dataframe data
        self.data_frame = ttk.Frame(self)
        self.data_frame.grid(row=3, column=0, columnspan=2, sticky='nsw')

        self.button_1 = ttk.Button(master=self, text="View Graphs", command=self.view_graphs)
        self.button_1.grid(row=4, column=0, columnspan=4)
        self.button_2 = ttk.Button(master=self, text="Back", command=self.callback)
        self.button_2.grid(row=5, column=0, columnspan=4)

        # Display the student details
        self.display_student_details()

    def display_student_details(self):
        # Fetch the student data from the CouchDB
        print("adhd test details = ", self.adhd_test_details)
        self.student_data = self.backend_api.get_student(self.adhd_test_details['_id'])

        # Extract student details from the CouchDB entry
        name = self.adhd_test_details['fullname']
        age = self.adhd_test_details['age']
        gender = self.adhd_test_details['gender']
        email = self.adhd_test_details['email']

        # Add additional details
        additional_details = {'Name': name, 'Age': age, 'Gender': gender, 'Email': email}
        row = 0
        for column, value in additional_details.items():
            label = ttk.Label(master=self.details_frame, font=("Helvetica", 10, 'bold'), text=f"{column}:",
                              justify=tkinter.LEFT)
            value_label = ttk.Label(master=self.details_frame, font=("Helvetica", 10, 'italic'), text=f"{value}",
                                    justify=tkinter.LEFT)  # Set value to italic
            label.grid(row=row, column=0, pady=10, padx=10, sticky='nsw')
            value_label.grid(row=row, column=1, pady=10, padx=10, sticky='nsw')
            row += 1

    def get_dataframe_data(self, student_name, session_type):
        # Map session type to numeric value
        session_type_map = {"Without Disturbances": 0, "With Disturbances": 1}
        session_type_num = session_type_map[session_type]

        # Connect to the CouchDB server
        couch = Server('http://admin:1234@localhost:5984/')

        # Access the 'test_dataframes' database
        db = couch['test_dataframes']

        # Initialize an empty DataFrame
        df = pd.DataFrame()

        # Iterate over all documents in the database
        for doc_id in db:
            # Get the document
            doc = db[doc_id]
            print("names = ", doc['Name'].values())
            # Check if the document contains the student's name and the selected session type
            if student_name in doc['Name'].values() and session_type_num in doc['Session Type'].values():
                # Convert the document to a DataFrame and append it to df

                doc_df = pd.DataFrame(doc)
                df = pd.concat([df, doc_df], ignore_index=True)

        # Filter the DataFrame to get the record where Name equals the student's name and Session Type equals the
        # selected session type
        student_record = df[(df['Name'] == student_name) & (df['Session Type'] == session_type_num)]

        # Check if student_record is empty
        if not student_record.empty:
            # Convert the filtered dataframe to a dictionary and return the first record
            return student_record.to_dict('records')[0]
        else:
            return None

    def display_dataframe_data(self, event=None):
        # Clear the data frame
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        print(self.student_data['fullname'])
        # Fetch the corresponding data from the dataframe
        self.dataframe_data = self.get_dataframe_data(
            self.student_data['adhd_test_details']['Individual']['Name'],
            self.session_type.get())
        # Check if dataframe_data is None
        if self.dataframe_data is None:
            # Handle the situation here. For example, you can show an error message and return:
            print("Error: No data found for the selected student and session type.")
            return

        # Use the dataframe_data as the student record
        student_record = self.dataframe_data

        # Create a label for each item in the DataFrame
        num_columns = len(student_record)  # Get the number of columns
        row = 0  # Start from the first row
        map_ADHD = {0: "No", 1: "Suspected"}
        map_meds = {0: "No", 1: "Yes"}
        map_sess = {0: "Without Disturbances", 1: "With Disturbances"}
        for column_name, value in student_record.items():
            if column_name not in ['_id', '_rev', 'Filename', 'Name']:  # Exclude the _id and _rev fields
                label = ttk.Label(master=self.data_frame, font=("Helvetica", 10, 'bold'), text=f"{column_name}:",
                                  justify=tkinter.LEFT)  # Set column name to bold
                if column_name == 'ADHD':
                    value_label = ttk.Label(master=self.data_frame, font=("Helvetica", 10, 'italic'),
                                            text=f"{map_ADHD[value]}",
                                            justify=tkinter.LEFT)  # Set value to italic
                elif column_name == 'Medication':
                    value_label = ttk.Label(master=self.data_frame, font=("Helvetica", 10, 'italic'),
                                            text=f"{map_meds[value]}",
                                            justify=tkinter.LEFT)  # Set value to italic
                elif column_name == 'Session Type':
                    value_label = ttk.Label(master=self.data_frame, font=("Helvetica", 10, 'italic'),
                                            text=f"{map_sess[value]}",
                                            justify=tkinter.LEFT)  # Set value to italic
                else:
                    value_label = ttk.Label(master=self.data_frame, font=("Helvetica", 10, 'italic'),
                                            text=f"{value}",
                                            justify=tkinter.LEFT)  # Set value to italic
                label.grid(row=row, column=0, pady=10, padx=10, sticky='nsw')
                value_label.grid(row=row, column=1, pady=10, padx=10, sticky='nsw')
                row += 1

    def view_graphs(self):
        self.parent.switch_frame(viewAnlyzeGraph, parent_frame=self.parent_frame, callback=self.back,
                                 student_data=self.student_data, dataframe_data=self.dataframe_data)

    def back(self):
        if self.callback:
            self.callback()

