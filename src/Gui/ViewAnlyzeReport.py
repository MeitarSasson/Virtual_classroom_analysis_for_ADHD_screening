from Gui.ViewAnlyzeGraph import viewAnlyzeGraph
from main import PaddedFrame, BasePage
import couchdb
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Import ttk


class ViewAnalyzedReportsFrame(PaddedFrame):
    def __init__(self, parent, parent_frame=None, callback=None, file=None):
        super().__init__(parent)
        self.parent_frame = parent_frame
        self.callback = callback
        self.file = file
        self.parent = parent
        self.label_5 = ttk.Label(self, font=("Helvetica", 20), text="View Report", justify=tk.CENTER)
        self.label_5.grid(row=0, column=0)

        # Connect to CouchDB
        couch = couchdb.Server('http://admin:1234@localhost:5984/')  # Replace with your CouchDB URL

        # Access the 'test_results' database
        self.db = couch['test_results']

        # Fetch all document IDs
        self.doc_ids = [row.id for row in self.db.view('_all_docs')]

        # Create a StringVar to hold the selected document ID
        self.selected_doc_id = tk.StringVar(value="")  # Set initial value to an empty string

        # Create a radio button for each document ID
        for i, doc_id in enumerate(self.doc_ids):
            rb = tk.Radiobutton(self, text=doc_id, variable=self.selected_doc_id, value=doc_id)
            rb.grid(row=i + 1, column=0)

        self.button_select_file = ttk.Button(self, text="Select File", command=self.select_file_and_display_report)
        self.button_select_file.grid(row=len(self.doc_ids) + 1, column=0)

        self.back_button = ttk.Button(self, text="Back", command=self.go_back)
        self.back_button.grid(row=len(self.doc_ids) + 1, column=1)

    def go_back(self):
        if self.callback:
            self.callback()

    def select_file_and_display_report(self):
        # Check if a radio button has been selected
        if not self.selected_doc_id.get():
            messagebox.showerror("File required", "Please select a file.")
            return

        # Fetch the selected document from the database
        doc = self.db[self.selected_doc_id.get()]

        # Pass the document to the next frame
        self.parent.switch_frame(viewAnlyzeReport, parent_frame=self.parent_frame,
                                 callback=self.callback, file=doc)


class viewAnlyzeReport(PaddedFrame):
    def __init__(self, parent, parent_frame, callback=None, file=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent_frame = parent_frame
        self.callback = callback
        self.parent = parent
        self.file = file
        self.label_2 = ttk.Label(master=self, font=("Helvetica", 30), text="Analyze Report",  # Use ttk.Label
                                 justify=tk.LEFT)
        self.label_2.grid(row=0, column=0, pady=10, padx=10)

        self.button_1 = ttk.Button(master=self, text="View Graph", command=self.view_analyze_graph)  # Use ttk.Button
        self.button_1.grid(row=6, column=0, pady=10, padx=10)

        self.button_2 = ttk.Button(master=self, text="Back", command=self.back)  # Use ttk.Button
        self.button_2.grid(row=7, column=0, pady=10, padx=10)

    def view_analyze_graph(self):
        file = self.file  # Use the selected file directly
        # Switch to the viewAnalyzeGraph frame
        self.parent.switch_frame(viewAnlyzeGraph, parent_frame=self.parent_frame,
                                 callback=self.back, file=file)

    def back(self):
        if self.callback:
            self.callback()
