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
    solution_1 = modelos.solve_one_dimensional_diffusion(area=0.01, L=0.5, d_x=0.1, Ta=100, Tb=500, k=1000)
    print(solution_1)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
