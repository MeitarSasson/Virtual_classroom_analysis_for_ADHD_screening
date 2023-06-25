import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from ttkthemes import ThemedTk

from Gui.DeveloperPage import developerPage
from Gui.DiagnoserPage import diagnoserPage
from Gui.StudentDetails import studentDetails, StudentRegister, StudentLogin
from main import BasePage, PaddedFrame


class loginPage(PaddedFrame):
    def __init__(self, master, frame_master=None, model=None, callback=None, **kwargs):
        kwargs.pop('parent', None)  # Remove the 'parent' key from kwargs
        super().__init__(master)
        self.master = master
        self.model = model
        self.callback = callback  # Set the callback attribute
        self.grid(sticky='nsew')  # use grid instead of pack
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.label_1 = ttk.Label(self, font=("Helvetica", 20), text="Welcome!", justify=tk.LEFT)
        self.label_1.grid(row=0, column=0, pady=20)  # change pack to grid

        self.combobox_1 = ttk.Combobox(self, values=["Student", "Diagnoser", "Developer"])
        self.combobox_1.grid(row=1, column=0, columnspan=2, pady=20)  # change pack to grid
        self.combobox_1.set("Choose User")

        self.button_1 = ttk.Button(self, text="Enter", command=self.Enter_User)
        self.button_1.grid(row=2, column=0, columnspan=2)  # change pack to grid

        self.button_2 = ttk.Button(self, text="Exit", command=self.Log_Out)
        self.button_2.grid(row=3, column=0, columnspan=2)  # change pack to grid

    def Enter_User(self):
        selected_actor = self.combobox_1.get()
        valid_actors = ["Student", "Diagnoser", "Developer"]
        if selected_actor not in valid_actors:
            messagebox.showerror("Error", "Please choose an actor before signing in")
        elif selected_actor == "Student":
            self.master.switch_frame(studentDetails, parent_frame=self.master, callback=self.master.switch_frame)
        elif selected_actor == "Diagnoser":
            self.master.switch_frame(diagnoserPage, parent=self.master, callback=self.master.switch_frame)
        elif selected_actor == "Developer":
            self.master.switch_frame(developerPage, parent=self.master, callback=self.master.switch_frame)

    def Log_Out(self):
        self.master.quit()
