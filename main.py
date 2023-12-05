# This is a sample Python script.
import modelos
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # print_hi('PyCharm')
    A = modelos.solve_one_dimensional_diffusion(0.01, 0.5, 0.1, 100, 500, 1000)
    print(A)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
