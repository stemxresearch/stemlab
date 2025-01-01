def difference_formulas(file_path, order="\\'"):
    # List to store lines matching the criteria
    lines_latex, lines_maths = [], []
    file_folder = r'K:\Backup\EE\APHa\3. Mathematics\10. Numerical differentiation\3. Derivative formulas - data'
    with open(f'{file_folder}/{file_path}', 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if '\\color{#01B3D1}{f' + order + '(x_{0}) =' in line:
                modified_line = line.split('=')[1].split("\\\hspace{1cm}")[0]
                latex = f"f{order}(x) = {modified_line.strip()}"
                maths = latex.replace("\\\dfrac{", "(").replace("}{", ") / (").replace("_{", "").replace("}","").replace("\\\,", "*")
                maths = f'{maths.replace("(h", "h")})'.replace("^{", "^(").replace(order, "").replace("^(*iv", "")
                maths = maths.replace(" \\\\hspace{0.5cm \\\\cdots \\\\hspace{0.5cm (1) $ </p>')", "")
                lines_maths.append(maths)
                lines_latex.append(f'\\({latex}\\)')
              
    for line in lines_maths:
        print(line)

    print("")
    for line in lines_latex:
        print(line)
        

difference_formulas(file_path='2. first derivative\Derivative1.py', order="\\'")
difference_formulas(file_path='3. second derivative\Derivative2.py', order="\\'\\'")
difference_formulas(file_path='4. third derivative\Derivative3.py', order="\\'\\'\\'")
difference_formulas(file_path='5. fourth derivative\Derivative4.py', order="^{\\\,iv}")