import matplotlib.pyplot as plt
import numpy as np

# 不使用LaTeX，使用matplotlib内置数学渲染
plt.rcParams['font.size'] = 14

# 创建图形
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# Esim公式 - 使用matplotlib的数学文本渲染
formula = r'$s_{Esim}(X,Y) = \frac{1}{n} \sum_{i=1}^{n} \omega_i \cdot e^{-\frac{|x_i-y_i|}{|x_i-y_i|+\frac{|x_i+y_i|}{2}}}$'

# 显示公式
ax.text(0.5, 0.5, formula, ha='center', va='center', fontsize=20, 
        transform=ax.transAxes, math_fontfamily='dejavusans')

# 保存图片
plt.tight_layout()
plt.savefig('esim_formula.png', dpi=300, bbox_inches='tight', transparent=True, 
            facecolor='white', edgecolor='none')
plt.close()

print('Esim公式图片已创建: esim_formula.png')
