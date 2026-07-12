import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_qe(filepath, plot, e_fermi=0.0, x_label="Cutoff Energy (Ry)", xlim=None, ylim=None, k_nodes=None, k_labels=None, save_as=None, c_supercell=None, t_effective=None):
    """
    The main function students will call.
    Note: For PDOS and Optics, 'filepath' acts as the file prefix (e.g., 'crsi2n4' or 'MoS2').
    """
    plot = plot.lower().strip()
    
    if plot == "bands":
        _plot_qe_bands(filepath, e_fermi, xlim, ylim, k_nodes, k_labels, save_as)
    elif plot == "dos":
        _plot_qe_dos(filepath, e_fermi, xlim, ylim, save_as)
    elif plot == "pdos":
        _plot_qe_pdos(filepath, e_fermi, xlim, ylim, save_as)
    elif plot == "convergence":
        _plot_convergence(filepath, x_label, xlim, ylim, save_as)
    elif plot == "optics":
        if c_supercell is None or t_effective is None:
            print("Error: For 2D optical properties, you must provide 'c_supercell' and 't_effective' parameters.")
        else:
            _plot_qe_optics(filepath, c_supercell, t_effective, xlim, save_as)
    else:
        print(f"Action '{plot}' not recognized.")


def _plot_qe_optics(prefix, c_supercell, t_effective, xlim, save_as=None):
    """
    Internal function to process and plot 2D optical properties in a 2x2 grid.
    Expects files named 'epsr_{prefix}.dat' and 'epsi_{prefix}.dat'.
    """
    try:
        print(f"Loading QE optics data for prefix '{prefix}'...")
        
        # Load real and imaginary dielectric data
        # Assuming QE standard output where column 1 (index 1) is the xx (in-plane) component
        data_r = np.loadtxt(f"epsr_{prefix}.dat")
        data_i = np.loadtxt(f"epsi_{prefix}.dat")
        
        energy = data_r[:, 0]
        eps1_calc = data_r[:, 1]
        eps2_calc = data_i[:, 1]
        
        # --- 2D SCALING CORRECTION ---
        scaling_factor = c_supercell / t_effective
        eps1_true = 1 + (eps1_calc - 1) * scaling_factor
        eps2_true = eps2_calc * scaling_factor
        
        # --- CALCULATIONS ---
        # 1. Complex magnitude
        eps_mag = np.sqrt(eps1_true**2 + eps2_true**2)
        # 2. Refractive Index (n)
        n = np.sqrt((eps_mag + eps1_true) / 2.0)
        # 3. Extinction Coefficient (k)
        k = np.sqrt((eps_mag - eps1_true) / 2.0)
        # 4. Absorption Coefficient alpha (cm^-1)
        alpha = 1.01e5 * energy * k
        # 5. Reflectivity (R)
        R = ((n - 1)**2 + k**2) / ((n + 1)**2 + k**2)

        # --- PLOTTING ---
        fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        
        # Plot 1: Dielectric Function
        axs[0, 0].plot(energy, eps1_true, color='#d62728', linewidth=2, label=r'$\varepsilon_1$ (Real)')
        axs[0, 0].plot(energy, eps2_true, color='#1f77b4', linewidth=2, label=r'$\varepsilon_2$ (Imaginary)')
        axs[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axs[0, 0].set_ylabel('Dielectric Function', fontsize=14)
        axs[0, 0].legend(loc='best')
        
        # Plot 2: Absorption Coefficient
        axs[0, 1].plot(energy, alpha, color='purple', linewidth=2)
        axs[0, 1].set_ylabel(r'Absorption Coefficient (cm$^{-1}$)', fontsize=14)
        
        # Plot 3: Refractive Index
        axs[1, 0].plot(energy, n, color='green', linewidth=2)
        axs[1, 0].set_ylabel('Refractive Index (n)', fontsize=14)
        
        # Plot 4: Reflectivity
        axs[1, 1].plot(energy, R, color='darkorange', linewidth=2)
        axs[1, 1].set_ylabel('Reflectivity (R)', fontsize=14)
        
        # Format all subplots consistently
        for ax in axs.flat:
            ax.set_xlabel('Energy (eV)', fontsize=14)
            ax.tick_params(axis='both', direction='in', length=5, labelsize=12)
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim(0, max(energy))
                
        plt.tight_layout(pad=2.0)

        # --- SAVING ---
        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"-> Plot successfully saved as '{save_as}'")

        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find 'epsr_{prefix}.dat' or 'epsi_{prefix}.dat'. Check your prefix and folder.")
    except Exception as e:
        print(f"An error occurred while plotting optics: {e}")


def _plot_qe_bands(filepath, e_fermi, xlim, ylim, k_nodes, k_labels, save_as):
    """
    Internal function using robust blank-line parsing and high-symmetry nodes.
    """
    try:
        print(f"Loading QE band data from {filepath}...")
        
        bands = []
        current_band = []
        
        # 1. ROBUST PARSING
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line: # If not empty
                    parts = line.split()
                    if len(parts) >= 2:
                        current_band.append([float(parts[0]), float(parts[1])])
                else: # Empty line means end of band
                    if current_band:
                        bands.append(np.array(current_band))
                        current_band = []
                        
        if current_band: # Catch the final band
            bands.append(np.array(current_band))

        # 2. PLOTTING
        plt.figure(figsize=(6, 8), dpi=300)
        
        for band in bands:
            k_dist = band[:, 0]
            energy = band[:, 1] - e_fermi
            plt.plot(k_dist, energy, color='#004c99', linewidth=2)

        # 3. HIGH-SYMMETRY POINTS
        if k_nodes is not None:
            for k in k_nodes:
                plt.axvline(x=k, color='black', linewidth=0.8, linestyle='-')
            if k_labels is not None:
                plt.xticks(k_nodes, k_labels, fontsize=14)
            # Default to tightly bounding the x-axis to the nodes
            plt.xlim(k_nodes[0], k_nodes[-1])

        plt.axhline(y=0, color='red', linewidth=1.2, linestyle='--', label='Fermi Level')

        # 4. FORMATTING
        plt.ylabel(r'$E - E_F$ (eV)', fontsize=14)
        plt.title('Electronic Band Structure', fontsize=16, pad=15)
        plt.tick_params(axis='x', direction='in', length=5)
        plt.tick_params(axis='y', direction='in', length=5)
        plt.yticks(fontsize=12)

        # Apply custom limits if provided by the student
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()

        # 5. SAVING
        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"-> Plot successfully saved as '{save_as}'")

        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while plotting: {e}")

def _plot_qe_dos(filepath, e_fermi, xlim, ylim, save_as=None):
    """
    Internal function to read QE .dos files and plot them.
    """
    try:
        print(f"Loading QE DOS data from {filepath}...")
        
        data = np.loadtxt(filepath)
        energies = data[:, 0]
        dos = data[:, 1]
        
        shifted_energies = energies - e_fermi
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(shifted_energies, dos, color='blue', linewidth=1.5)
        
        plt.axvline(0, color='red', linestyle='--', label='Fermi Level (0 eV)')
        
        plt.fill_between(shifted_energies, 0, dos, where=(shifted_energies <= 0), 
                         color='gray', alpha=0.3)
        
        plt.ylabel('Density of States', fontsize=14)
        plt.xlabel('Energy (eV)', fontsize=14)
        plt.title('Electronic Density of States', fontsize=16)
        
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(-5, 5)
            
        if ylim is not None:
            plt.ylim(ylim)
        
        plt.legend()

        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"-> Plot successfully saved as '{save_as}'")

        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while plotting DOS: {e}")

def _plot_qe_pdos(prefix, e_fermi, xlim, ylim, save_as=None):
    """
    Internal function to plot Total DOS alongside grouped Element + Orbital PDOS.
    """
    try:
        print(f"Searching for PDOS files with prefix '{prefix}'...")
        
        # 1. Find and plot the TOTAL PDOS file first
        tot_pattern = f"{prefix}*.pdos_tot*"
        tot_files = glob.glob(tot_pattern)
        
        plt.figure(figsize=(8, 6), dpi=300)
        energies = None
        
        if tot_files:
            print(f"-> Found Total DOS file: {tot_files[0]}")
            tot_data = np.loadtxt(tot_files[0])
            energies = tot_data[:, 0] - e_fermi
            tot_dos = tot_data[:, 1]
            
            # Plot the Total DOS as a bold, solid black line
            plt.plot(energies, tot_dos, color='gray', linewidth=0.5, label='Total DOS')
            # Shade it lightly so the underlying orbitals are still visible
            plt.fill_between(energies, 0, tot_dos, color='gray', alpha=0.1)
        else:
            print("-> Warning: Total DOS file (*pdos_tot*) not found. Plotting only partials.")

        # 2. Find and process all the partial orbital files
        file_pattern = f"{prefix}*.pdos_atm*"
        pdos_files = glob.glob(file_pattern)
        
        if not pdos_files:
            print(f"Error: Could not find any partial files matching '{file_pattern}'.")
            return

        orbital_pdos = {}

        for file in pdos_files:
            match = re.search(r'atm#\d+\(([A-Za-z]+)\)_wfc#\d+\(([a-zA-Z])\)', file)
            if not match:
                continue
                
            element = match.group(1)
            orbital = match.group(2)
            label = f"{element} ({orbital})"

            data = np.loadtxt(file)

            if energies is None:
                energies = data[:, 0] - e_fermi

            ldos = data[:, 1]

            if label not in orbital_pdos:
                orbital_pdos[label] = np.zeros_like(ldos)
            
            orbital_pdos[label] += ldos

        # Plot each grouped orbital over the Total DOS
        for label, pdos in orbital_pdos.items():
            plt.plot(energies, pdos, linewidth=1.5, label=label)

        # 3. Formatting
        plt.axvline(0, color='red', linestyle='--', label='Fermi Level (0 eV)')

        plt.ylabel('Density of States', fontsize=14)
        plt.xlabel('Energy (eV)', fontsize=14)
        plt.title('Total and Projected Density of States', fontsize=16)

        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(-5, 5)
            
        if ylim is not None:
            plt.ylim(ylim)

        # Move the legend outside the plot area if there are many orbitals
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout() # Ensures the outside legend doesn't get cut off

        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"-> Plot successfully saved as '{save_as}'")

        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting PDOS: {e}")

def _plot_convergence(filepath, x_label, xlim, ylim, save_as=None):
    """
    Internal function to plot a 2-column text file (e.g., Cutoff vs Total Energy).
    """
    try:
        print(f"Loading convergence data from {filepath}...")
        
        data = np.loadtxt(filepath)
        parameter_vals = data[:, 0]
        total_energies = data[:, 1]
        
        plt.figure(figsize=(8, 6), dpi=300)
        
        plt.plot(parameter_vals, total_energies, marker='o', color='purple', 
                 linestyle='-', linewidth=1.5, markersize=8)
        
        plt.ylabel('Total Energy (Ry)', fontsize=14)
        plt.xlabel(x_label, fontsize=14)
        plt.title('Convergence Test', fontsize=16)
        
        # Add this line to force normal numbers instead of scientific offsets!
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if save_as:
            plt.savefig(save_as, dpi=300)
            print(f"-> Plot successfully saved as '{save_as}'")

        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")
    except Exception as e:
        print(f"An error occurred while plotting convergence: {e}")