"""
Line Ground Fault and Line Trip Module for ParaEMT
====================================================

This module implements transmission line ground fault simulation with three phases:
1. Pre-fault: Line is split at fault point with fault admittance = 0
2. During-fault: Fault admittance (e.g., 1e4) is applied to middle bus
3. Post-fault: Faulted line is removed from the system

Fault parameters:
- Line number (line_idx)
- Fault location (0.0-1.0, where 0=from_bus, 1=to_bus)
- Fault occurrence time (t_fault)
- Fault clearing time (t_fault_clear)
- Fault resistance (R_fault)
- Fault conductance (G_fault, typically 1e4 for ground fault)

Author: ParaEMT Contributors
Date: 2025
License: BSD-3-Clause
"""

import numpy as np
import scipy.sparse as sp


class LineFaultConfig:
    """Configuration for a single line fault event"""
    
    def __init__(self):
        """Initialize line fault parameters"""
        # Line identification
        self.line_idx = None          # Index in line_from/line_to arrays
        self.line_bus_from = None     # From bus number
        self.line_bus_to = None       # To bus number
        self.line_id = '1'            # Line circuit ID
        
        # Fault location and timing
        self.fault_location = 0.5     # Fault location: 0.0-1.0 (0=from bus, 1=to bus)
        self.t_fault = 1.0            # Time when fault occurs (seconds)
        self.t_fault_clear = 1.2      # Time when fault is cleared (seconds)
        
        # Fault parameters
        self.fault_resistance = 0.0   # Fault resistance (ohms)
        self.fault_conductance = 1e4  # Fault admittance (1/ohms) during fault (typically 1e4)
        
        # Internal state tracking
        self.middle_bus_idx = None    # Index of newly created middle bus
        self.middle_bus_num = None    # Bus number of middle bus
        self.fault_phase = 0          # 0: pre-fault, 1: during-fault, 2: post-fault
        self.flag_fault_initialized = False
        self.flag_fault_active = False
        self.flag_fault_cleared = False
        
        # Store original line segment indices for removal
        self.line_segment1_idx = None
        self.line_segment2_idx = None


class LineFaultManager:
    """Manager for handling multiple line fault events"""
    
    def __init__(self):
        """Initialize the line fault manager"""
        self.faults = []              # List of LineFaultConfig objects
        self.nbus_original = None     # Original number of buses
        self.nbus_with_faults = None  # Number of buses after adding fault points
        self.fault_bus_mapping = {}   # Map from line_idx to middle bus index
        self.lines_to_remove = set()  # Lines to remove after fault clearing
        
    def add_fault(self, fault_config):
        """Add a new line fault configuration"""
        if fault_config.line_idx is None:
            raise ValueError("line_idx must be specified in fault configuration")
        self.faults.append(fault_config)
        
    def validate_config(self):
        """Validate all fault configurations"""
        for fault in self.faults:
            if fault.line_idx is None:
                raise ValueError("line_idx must be specified")
            if not (0.0 <= fault.fault_location <= 1.0):
                raise ValueError("fault_location must be between 0.0 and 1.0")
            if fault.t_fault_clear <= fault.t_fault:
                raise ValueError("t_fault_clear must be > t_fault")
            if fault.fault_conductance < 0:
                raise ValueError("fault_conductance must be non-negative")


def split_line_at_fault_point(line_RX, fault_location):
    """
    Split a line impedance at the fault point into two segments.
    
    The original line impedance is split proportionally based on fault location.
    This ensures Z_total = Z1 + Z2 for accurate fault representation.
    
    Parameters:
    -----------
    line_RX : complex
        Original line impedance (R + jX)
    fault_location : float
        Fault location relative to line length (0.0 to 1.0)
        
    Returns:
    --------
    z_segment1 : complex
        Impedance of segment 1 (from-bus to fault point)
    z_segment2 : complex
        Impedance of segment 2 (fault point to to-bus)
        
    Notes:
    ------
    For numerical stability:
    - Real part (R) is split linearly
    - Imaginary part (X) is split linearly
    - Verification: Z1 + Z2 = Z_original
    """
    # Split impedance proportionally
    z_segment1 = line_RX * fault_location
    z_segment2 = line_RX * (1.0 - fault_location)
    
    return z_segment1, z_segment2


def split_line_charging(line_chg, fault_location):
    """
    Split line charging susceptance at fault point.
    
    The charging susceptance is split to represent the equivalent two-section line.
    For a PI-model line, charging is placed at both ends.
    
    Parameters:
    -----------
    line_chg : float
        Original line charging susceptance
    fault_location : float
        Fault location (0.0 to 1.0)
        
    Returns:
    --------
    chg_segment1 : float
        Charging susceptance of segment 1
    chg_segment2 : float
        Charging susceptance of segment 2
        
    Notes:
    ------
    Charging is proportionally split. For more accuracy, consider
    the pi-model shunt placement at middle bus.
    """
    # Split charging proportionally
    chg_segment1 = line_chg * fault_location
    chg_segment2 = line_chg * (1.0 - fault_location)
    
    return chg_segment1, chg_segment2


def calculate_line_admittance(line_RX, ts=50e-6, ws=2*np.pi*60):
    """
    Calculate equivalent admittance for a line segment using Trapezoidal discretization.
    
    This uses the Trapezoidal rule for numerical integration of the line model,
    consistent with ParaEMT's EMT formulation.
    
    Parameters:
    -----------
    line_RX : complex
        Line impedance (R + jX)
    ts : float
        Time step (default: 50e-6 s for 20 kHz)
    ws : float
        Angular frequency (default: 2*pi*60 rad/s for 60 Hz)
        
    Returns:
    --------
    Y_eq : complex
        Equivalent admittance for the discretized model
        
    Notes:
    ------
    The implementation follows the discretization in lib_numba.py:
    - Inductive line: Req = (1 + R*(ts/2L + 1/Rp)) / (ts/2L + 1/Rp)
    - Capacitive line: Req = R + ts/2CL
    - Damping resistor: Rp = (20/3) * (2L/ts)
    """
    R = np.real(line_RX)
    X = np.imag(line_RX)
    
    if X > 0:  # Inductive line
        L = X / ws
        # Trapezoidal discretization with damping
        damptrap = 1
        Rp = damptrap * (20.0 / 3.0) * (2.0 * L / ts)
        Rp_inv = 1.0 / Rp if Rp != 0 else 1e-10
        
        Req = (1.0 + R * (ts / (2.0 * L) + Rp_inv)) / (ts / (2.0 * L) + Rp_inv)
        Y_eq = 1.0 / Req if Req != 0 else 1e10
        
    elif X < 0:  # Capacitive line
        CL = -1.0 / (X * ws)
        Req = R + ts / (2.0 * CL)
        Y_eq = 1.0 / Req if Req != 0 else 1e10
        
    else:  # Pure resistance
        Y_eq = 1.0 / R if R != 0 else 1e10
        
    return Y_eq


class FaultAdmittanceUpdater:
    """
    Utility class for updating admittance matrix during fault evolution.
    
    This class manages dynamic updates to the system admittance matrix G0
    for:
    1. Adding fault conductance to ground (fault occurs)
    2. Removing fault conductance (fault cleared)
    3. Removing line segments (line trip)
    
    The admittance matrix is stored in LIL (List of Lists) format for
    efficient element-level modifications.
    """
    
    def __init__(self, G0_coo, nbus):
        """
        Initialize the updater with the initial admittance matrix.
        
        Parameters:
        -----------
        G0_coo : scipy.sparse.coo_matrix
            Initial system admittance matrix in COO format
            Shape: (3*nbus, 3*nbus) for three-phase system
        nbus : int
            Number of buses in the original system
        """
        self.G0_lil = G0_coo.tolil()  # Convert to LIL for efficient updates
        self.nbus = nbus
        self.fault_admittances = {}   # Track added fault admittances: (bus_idx, phase) -> G_value
        
    def add_fault_to_ground(self, bus_idx, fault_conductance):
        """
        Add fault conductance to ground (shunt admittance) at specified bus.
        
        For a three-phase system, fault conductance is added to all three phases:
        Y_ground_a = Y_ground_b = Y_ground_c = fault_conductance
        
        Parameters:
        -----------
        bus_idx : int
            Bus index (0 to nbus-1)
        fault_conductance : float
            Fault conductance value (typically 1e4 for ground fault)
            
        Notes:
        ------
        Modifies the diagonal elements of admittance matrix:
        G[3*bus+phase, 3*bus+phase] += fault_conductance for phase in [0, 1, 2]
        """
        # Add to all three phases (A, B, C)
        for phase in range(3):
            row_idx = bus_idx + phase * self.nbus
            self.G0_lil[row_idx, row_idx] += fault_conductance
            self.fault_admittances[(bus_idx, phase)] = fault_conductance
            
    def remove_fault_from_ground(self, bus_idx, fault_conductance):
        """
        Remove fault conductance from ground at specified bus.
        
        Parameters:
        -----------
        bus_idx : int
            Bus index
        fault_conductance : float
            Fault conductance value to remove
            
        Notes:
        ------
        This reverses the effect of add_fault_to_ground().
        """
        for phase in range(3):
            row_idx = bus_idx + phase * self.nbus
            self.G0_lil[row_idx, row_idx] -= fault_conductance
            if (bus_idx, phase) in self.fault_admittances:
                del self.fault_admittances[(bus_idx, phase)]
                
    def remove_line_segment(self, from_bus_idx, to_bus_idx, Y_line):
        """
        Remove line admittance from the system (line tripping).
        
        When a line is removed, its admittance contribution must be subtracted
        from the system admittance matrix. For a line connecting bus i to j:
        
        Before removal:
          Y[i,i] += Y_line,  Y[i,j] -= Y_line
          Y[j,i] -= Y_line,  Y[j,j] += Y_line
          
        After removal (undo the above):
          Y[i,i] -= Y_line,  Y[i,j] += Y_line
          Y[j,i] += Y_line,  Y[j,j] -= Y_line
        
        Parameters:
        -----------
        from_bus_idx : int
            From bus index
        to_bus_idx : int
            To bus index
        Y_line : complex
            Line admittance to remove
        """
        # Remove contributions from all three phases
        for phase in range(3):
            r_from = from_bus_idx + phase * self.nbus
            r_to = to_bus_idx + phase * self.nbus
            
            # Remove diagonal terms (reduce self-admittance)
            self.G0_lil[r_from, r_from] -= np.real(Y_line)
            self.G0_lil[r_to, r_to] -= np.real(Y_line)
            
            # Remove off-diagonal terms (set mutual admittance to zero)
            self.G0_lil[r_from, r_to] += np.real(Y_line)
            self.G0_lil[r_to, r_from] += np.real(Y_line)
                
    def get_updated_matrix_coo(self):
        """
        Return updated admittance matrix in COO format for factorization.
        
        Returns:
        --------
        G_updated : scipy.sparse.coo_matrix
            Updated system admittance matrix
        """
        return self.G0_lil.tocoo()
    
    def get_updated_matrix_lil(self):
        """
        Return updated admittance matrix in LIL format for further modifications.
        
        Returns:
        --------
        G_updated : scipy.sparse.lil_matrix
            Updated system admittance matrix
        """
        return self.G0_lil


class LineFaultManager_Extended(LineFaultManager):
    """
    Extended manager with numerical methods for fault handling.
    
    This class handles the full lifecycle of line faults in ParaEMT:
    1. Preparation: Split lines and add middle buses before simulation
    2. Initialization: Update network admittance matrix
    3. Simulation: Manage fault phase transitions
    """
    
    def prepare_pfd_for_faults(self, pfd):
        """
        Prepare power flow data structure for line faults by adding middle buses.
        
        This method is called ONCE during initialization BEFORE simulation starts.
        It:
        1. Creates middle buses for each fault location
        2. Splits original lines into two segments
        3. Updates pfd.bus_num, pfd.line_from, pfd.line_to, etc.
        
        Parameters:
        -----------
        pfd : PFData
            Power flow data object (modified in-place)
            
        Notes:
        ------
        After calling this method, you must call:
        - ini.InitNet(pfd, ts, loadmodel_option) to rebuild admittance matrix
        - ini.MergeMacG(pfd, dyd, ts, i_gentrip, netMod) for generator coupling
        """
        self.nbus_original = len(pfd.bus_num)
        
        # Track new buses and lines to add
        new_buses = []
        new_lines = []
        
        # Process each fault
        for fault_idx, fault in enumerate(self.faults):
            # Create middle bus with unique number
            new_bus_num = int(max(pfd.bus_num)) + fault_idx + 1
            
            # Get voltage base from from_bus
            from_bus_idx = np.where(pfd.bus_num == pfd.line_from[fault.line_idx])[0][0]
            
            new_buses.append({
                'bus_num': new_bus_num,
                'bus_type': 1,  # PQ bus
                'bus_Vm': 1.0,
                'bus_Va': 0.0,
                'bus_kV': pfd.bus_kV[from_bus_idx],
                'bus_basekV': pfd.bus_basekV[from_bus_idx],
                'bus_name': f'FAULT_BUS_{new_bus_num}'
            })
            
            # Get original line data
            orig_line_idx = fault.line_idx
            line_from = pfd.line_from[orig_line_idx]
            line_to = pfd.line_to[orig_line_idx]
            line_RX = pfd.line_RX[orig_line_idx]
            line_chg = pfd.line_chg[orig_line_idx]
            
            # Store for later reference
            fault.line_bus_from = line_from
            fault.line_bus_to = line_to
            fault.middle_bus_num = new_bus_num
            
            # Split line impedance and charging
            z_seg1, z_seg2 = split_line_at_fault_point(line_RX, fault.fault_location)
            chg_seg1, chg_seg2 = split_line_charging(line_chg, fault.fault_location)
            
            # Create two new line segments
            new_lines.append({
                'from_bus': line_from,
                'to_bus': new_bus_num,
                'RX': z_seg1,
                'charging': chg_seg1,
                'line_id': fault.line_id + '_1'
            })
            
            new_lines.append({
                'from_bus': new_bus_num,
                'to_bus': line_to,
                'RX': z_seg2,
                'charging': chg_seg2,
                'line_id': fault.line_id + '_2'
            })
            
            # Store middle bus index
            fault.middle_bus_idx = self.nbus_original + fault_idx
            self.fault_bus_mapping[orig_line_idx] = fault.middle_bus_idx
            fault.flag_fault_initialized = True
        
        # Add new buses to pfd
        for new_bus in new_buses:
            pfd.bus_num = np.append(pfd.bus_num, new_bus['bus_num'])
            pfd.bus_type = np.append(pfd.bus_type, new_bus['bus_type'])
            pfd.bus_Vm = np.append(pfd.bus_Vm, new_bus['bus_Vm'])
            pfd.bus_Va = np.append(pfd.bus_Va, new_bus['bus_Va'])
            pfd.bus_kV = np.append(pfd.bus_kV, new_bus['bus_kV'])
            pfd.bus_basekV = np.append(pfd.bus_basekV, new_bus['bus_basekV'])
            pfd.bus_name = np.append(pfd.bus_name, new_bus['bus_name'])
        
        # Mark original faulted lines for removal
        lines_to_remove = set([fault.line_idx for fault in self.faults])
        self.lines_to_remove = lines_to_remove
        
        # Create new line arrays
        new_line_from = []
        new_line_to = []
        new_line_RX = []
        new_line_chg = []
        new_line_id = []
        new_line_P = []
        new_line_Q = []
        line_idx_mapping = {}  # Old index -> new index
        
        # Keep non-faulted lines
        new_idx = 0
        for i in range(len(pfd.line_from)):
            if i not in lines_to_remove:
                new_line_from.append(pfd.line_from[i])
                new_line_to.append(pfd.line_to[i])
                new_line_RX.append(pfd.line_RX[i])
                new_line_chg.append(pfd.line_chg[i])
                new_line_id.append(pfd.line_id[i])
                new_line_P.append(pfd.line_P[i])
                new_line_Q.append(pfd.line_Q[i])
                line_idx_mapping[i] = new_idx
                new_idx += 1
        
        # Add new fault line segments
        for new_line in new_lines:
            new_line_from.append(new_line['from_bus'])
            new_line_to.append(new_line['to_bus'])
            new_line_RX.append(new_line['RX'])
            new_line_chg.append(new_line['charging'])
            new_line_id.append(new_line['line_id'])
            new_line_P.append(0.0)
            new_line_Q.append(0.0)
            new_idx += 1
        
        # Update pfd with new line data
        pfd.line_from = np.array(new_line_from)
        pfd.line_to = np.array(new_line_to)
        pfd.line_RX = np.array(new_line_RX)
        pfd.line_chg = np.array(new_line_chg)
        pfd.line_id = np.array(new_line_id)
        pfd.line_P = np.array(new_line_P)
        pfd.line_Q = np.array(new_line_Q)
        
        self.nbus_with_faults = len(pfd.bus_num)
