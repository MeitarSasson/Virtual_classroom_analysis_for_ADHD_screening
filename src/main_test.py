
from ttkthemes import ThemedTk
from Gui.DeveloperPage import developerPage
from Gui.DiagnoserPage import diagnoserPage
from Gui.LoginPage import loginPage
import unittest
from tkinter import Tk
from Gui.StudentDetails import studentDetails, StudentRegister, StudentLogin


class StudentDetailsPageTests(unittest.TestCase):
    def setUp(self):
        self.root = Tk()
        self.page = studentDetails(self.root, parent_frame=None)

    def tearDown(self):
        self.root.destroy()

    def test_switch_to_login_page(self):
        self.page.go_to_login()
        self.assertIsInstance(self.page.parent.frame, StudentLogin)

class LoginPageTests(unittest.TestCase):
    def setUp(self):
        self.root = ThemedTk(theme="arc")
        self.page = loginPage(self.root)

    def tearDown(self):
        self.root.destroy()

    def test_combobox_default_value(self):
        self.assertEqual(self.page.combobox_1.get(), "Choose User")

    def test_enter_student_user(self):
        self.page.combobox_1.set("Student")
        self.page.Enter_User()
        self.assertIsInstance(self.page.master.frame, studentDetails)

    def test_enter_diagnoser_user(self):
        self.page.combobox_1.set("Diagnoser")
        self.page.Enter_User()
        self.assertIsInstance(self.page.master.frame, diagnoserPage)

    def test_enter_developer_user(self):
        self.page.combobox_1.set("Developer")
        self.page.Enter_User()
        self.assertIsInstance(self.page.master.frame, developerPage)


if __name__ == '__main__':
    unittest.main()
