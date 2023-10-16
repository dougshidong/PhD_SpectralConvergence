import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import draw_loglog_slope as dls

import matplotlib.backends.backend_pdf

plt.rcParams['text.usetex'] = True

PLOT_ORDERS = False
PLOT_ORDERS = True

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

figure_size = (8, 6)
if PLOT_ORDERS:
    figure_size = (15, 8)
    width_ratios = [6, 2]

cases = {
    'naca0012_M050_A200': { 'input_file': './naca0012_subsonic.data',
                           'output_file': 'naca0012_subsonic_verification.pdf',
                           'p_range': range(4),
                           'p_reference': 3,
                           'origin_drag' : {
                                    0: (6.5e-3, 2e-2),
                                    1: (4.5e-3, 3e-3),
                                    2: (2.5e-3, 3e-4),
                                    3: (1.5e-3, 5e-5),
                                    4: (1.5e-3, 1e-5),
                                    },
                           'origin_lift' : {
                                    0: (6.5e-3, 4e-2),
                                    1: (3.5e-3, 9e-3),
                                    2: (2.5e-3, 5e-4),
                                    3: (1.9e-3, 2e-5),
                                    4: (1.5e-3, 1e-5),
                                }
                             },
    'naca0012_M085_A125': { 'input_file': './naca0012_transonic.data',
                           'output_file': 'naca0012_transonic_verification.pdf', 
                           'p_range': range(4),
                           'p_reference': 3,
                           'origin_drag' : {
                                    0: (6.5e-3, 2e-2),
                                    1: (4.5e-3, 3e-3),
                                    2: (2.5e-3, 3e-4),
                                    3: (1.5e-3, 5e-5),
                                    4: (1.5e-3, 1e-5),
                                    },
                           'origin_lift' : {
                                    0: (6.5e-3, 1e-2),
                                    1: (3.5e-3, 9e-2),
                                    2: (2.5e-3, 5e-3),
                                    3: (1.9e-3, 2e-3),
                                    4: (1.5e-3, 1e-5),
                                }
                             },
}

                           
current_case = cases['naca0012_M050_A200']
current_case = cases['naca0012_M085_A125']

def richardson_extrapolation(f1, f2, h1, h2, p):
    print(f1, f2)
    print(h1, h2)
    r = h1 / h2
    #return f1 + ((f1 - f2) / (r**(p) - 1.0))
    return (f1 - f2 * r**(p+1)) / (1.0-r**(p+1))

df = pd.read_csv(current_case['input_file'], delim_whitespace=True)
pdfName = current_case['output_file']
p_range = current_case['p_range']
p_reference = current_case['p_reference']
origin_drag = current_case['origin_drag']
origin_lift = current_case['origin_lift']
pdf = matplotlib.backends.backend_pdf.PdfPages(pdfName)




if PLOT_ORDERS:
    gs = gridspec.GridSpec(nrows=len(p_range), ncols=2, width_ratios=width_ratios)
else:
    gs = gridspec.GridSpec(nrows=1, ncols=1)


def plot_error_convergence(ykey, ylabel, origin_triangles):

    f1 = df[df['p'] == p_reference][ykey].values[-1]
    f2 = df[df['p'] == p_reference][ykey].values[-2]
    h1 = df[df['p'] == p_reference]['one_sqrt_dof'].values[-1]
    h2 = df[df['p'] == p_reference]['one_sqrt_dof'].values[-2]
    val_ref = richardson_extrapolation(f1, f2, h1, h2, p_reference)
    print('Reference %s: %f' % (ykey, val_ref))

    fig = plt.figure(figsize=figure_size)
    ax_main = plt.subplot(gs[0:,0])
    rate = {}
    axes = {}
    if PLOT_ORDERS:
        for i, p in enumerate(p_range):
            axes[i] = fig.add_subplot(gs[i,1])

            value_error = abs(df[df['p'] == p][ykey]-val_ref).to_numpy()
            value_error_ratio = np.divide(value_error[1:], value_error[:-1])
            h = df[df['p'] == p]['one_sqrt_dof'].to_numpy()
            h_ratio = np.divide(h[1:], h[:-1])
            rate[i] = np.log(value_error_ratio) / np.log(h_ratio)

    for i, p in enumerate(p_range):
        df_p = df[df['p'] == p]
        ax_main.loglog(df_p['one_sqrt_dof'], abs(df_p[ykey]-val_ref), "-o", color=colors[i], label=r'$p = %d$' % (p))
        dls.draw_loglog_slope(fig, ax_main, origin_triangles[p], 0.5, p+1, inverted=True)

    xlim_main = ax_main.get_xlim()
    for i, p in enumerate(p_range):
        if PLOT_ORDERS:
            df_p = df[df['p'] == p]
            axes[i].semilogx(df_p['one_sqrt_dof'][1:], rate[i], "-o", color=colors[i], label=r'$p = %d$' % (p))
            lower_ylim = np.floor(min(rate[i]))
            upper_ylim = np.ceil(max(rate[i]))
            if upper_ylim - lower_ylim <= 1:
                lower_ylim -= 1
                upper_ylim += 1
            axes[i].set_ylim([lower_ylim, upper_ylim])
            axes[i].set_xlim(xlim_main)
            axes[i].set_ylabel(r'Rate, $p = %d$' % (p) )
            axes[i].grid(True, which="major", ls="-", alpha=0.5)

            # Plot an average line
            #axes[i].axhline(y=np.nanmean(rate[i]), color='k', linestyle='--', linewidth=1.0)

    axes[len(p_range)-1].set_xlabel(r'$1 / \sqrt{DoFs}$')

    ax_main.legend()
    ax_main.set_ylabel(ylabel)
    ax_main.set_xlabel(r'$1 / \sqrt{DoFs}$')
    
    # Set transparent gridlines
    ax_main.grid(True, which="major", ls="-", alpha=0.5)

    pdf.savefig(fig, bbox_inches='tight')

plot_error_convergence('drag', r'Drag Error', origin_drag)
plot_error_convergence('lift', r'Lift Error', origin_lift)

# f1 = df[df['p'] == p_reference]['lift'].values[-1]
# f2 = df[df['p'] == p_reference]['lift'].values[-2]
# h1 = df[df['p'] == p_reference]['one_sqrt_dof'].values[-1]
# h2 = df[df['p'] == p_reference]['one_sqrt_dof'].values[-2]
# #h1 = (1.0/(df[df['p'] == p_reference]['cells'].values[-1]))**(1.0/2.0)
# #h2 = (1.0/(df[df['p'] == p_reference]['cells'].values[-2]))**(1.0/2.0)
# lift_reference = richardson_extrapolation(f1, f2, h1, h2, p_reference)
# print('Reference lift: %f' % (lift_reference))
# #lift_reference = 0.2865136558
# 
# 
# fig = plt.figure(figsize=figure_size)
# ax_main = plt.subplot(gs[0:,0])
# if PLOT_ORDERS:
#     for i, p in enumerate(p_range):
#         axes[i] = fig.add_subplot(gs[i,1])
# 
#         lift_error = abs(df[df['p'] == p]['lift']-lift_reference).to_numpy()
#         lift_error_ratio = np.divide(lift_error[1:], lift_error[:-1])
#         h = df[df['p'] == p]['one_sqrt_dof'].to_numpy()
#         h_ratio = np.divide(h[1:], h[:-1])
#         rate[i] = np.log(lift_error_ratio) / np.log(h_ratio)
# for i, p in enumerate(p_range):
#     df_p = df[df['p'] == p]
#     ax_main.loglog(df_p['one_sqrt_dof'], abs(df_p['lift']-lift_reference), "-o", color=colors[i], label=r'$p = %d$' % (p))
#     dls.draw_loglog_slope(fig, ax_main, origin_lift[p], 0.5, p+1, inverted=True)#False)
# 
#     if PLOT_ORDERS:
#         axes[i].semilogx(df_p['one_sqrt_dof'][1:], rate[i], "-o", color=colors[i], label=r'$p = %d$' % (p))
#         axes[i].set_ylim([np.floor(min(rate[i])), np.ceil(max(rate[i]))])
#         axes[i].set_ylabel(r'Rate, $p = %d$' % (p) )
# 
# axes[len(p_range)-1].set_xlabel(r'$1 / \sqrt{DoFs}$')
# 
# ax_main.legend()
# ax_main.set_ylabel(r'Lift Error')
# ax_main.set_xlabel(r'$1 / \sqrt{DoFs}$')
# pdf.savefig(fig, bbox_inches='tight')

pdf.close()