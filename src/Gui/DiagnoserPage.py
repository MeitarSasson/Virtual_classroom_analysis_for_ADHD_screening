import tkinter as tk
from tkinter import ttk, messagebox

from Gui.ViewReports import ViewReportsFrame
from main import PaddedFrame
from Gui.ViewAnlyzeReport import ViewAnalyzedReportsFrame


class diagnoserPage(PaddedFrame):
    def __init__(self, master, parent=None, parent_frame=None, model=None, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.callback = callback
        self.parent = parent
        self.master = master
        self.label_3 = ttk.Label(master=self, font=("Helvetica", 20), text="Welcome, Diagnoser!",
                                 justify=tk.LEFT)
        self.label_3.grid(row=0, column=0, pady=10, padx=10)

        self.combobox_3 = ttk.Combobox(self, values=["View Individual Report", "View Group Report"])
        self.combobox_3.grid(row=1, column=0, columnspan=2, pady=20, padx=20)
        self.combobox_3.set("Choose Action")

        self.button_4 = ttk.Button(master=self, text="Continue", command=self.choose_action)
        self.button_4.grid(row=2, column=0, columnspan=2)

        self.button_5 = ttk.Button(master=self, text="Back", command=self.callback)
        self.button_5.grid(row=3, column=0, columnspan=2)

    def choose_action(self):
        selected_action = self.combobox_3.get()
        if not selected_action or selected_action == "Choose Action":
            messagebox.showerror("Action required", "Please choose an action from the dropdown menu")
            return

        if selected_action == "View Individual Report":
            self.master.switch_frame(ViewAnalyzedReportsFrame,
                                     callback=self.master.switch_frame)  # Access the app instance using
            # self.parent.switch_frame
        elif selected_action == "View Group Report":
            self.master.switch_frame(ViewReportsFrame, parent=self.master, callback=self.master.switch_frame)

    def Log_Out(self):
        if self.callback:
            self.callback()
        self.destroy()
