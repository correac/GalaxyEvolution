import numpy as np
import commah
import matplotlib.pyplot as plt


def M200accr(M200, zinitial, zoutput, cosmo='planck15', commah_flag=True):
    """
    Function that calls the accretion routine from the COMMAH
    package, which calculates the accretion rate of a halo at any
    redshift 'z', given the halo total mass 'Mi' at redshift z.

    Parameters
    ----------
    M200 : float
    Logarithmic halo mass at redshift zinitial. Note, the halo is in
    units of M200 (mass enclosed within the virial radius, R200,
    defined as the radius at which the mean density is 200 times
    the critical).

    zinitial : float
    Redshift at which the halo has a mass M200.

    zoutput : float / array
    Redshift list at which the halo mass is calculated.

    cosmo : dict, optional.
    Dictionary of cosmological parameters, similar in format to:
    {'N_nu': 0,'Y_He': 0.24, 'h': 0.702, 'n': 0.963,'omega_M_0': 0.275,
    'omega_b_0': 0.0458,'omega_lambda_0': 0.725,'omega_n_0': 0.0,
    'sigma_8': 0.816, 't_0': 13.76, 'tau': 0.088,'z_reion': 10.6}.
    Default: planck15 (cosmological parameters of Planck15).

    Returns
    -------
    M(z) : float / array
    Halo mass [Msol] at redshift 'zoutput'.
    """
    if commah_flag:
        commah_run = commah.run(cosmo, zinitial, 10 ** M200, zoutput)
        output = commah_run['Mz'].flatten()
    else:
        x = M200 - 12.0
        if z <= 4.0:
            a = 0.83 + 0.553 * z - 0.0523 * z ** 2
            b = 1.436 - 0.149 * z + 0.007 * z ** 2
            c = -0.134 + 0.099 * z - 0.023 * z ** 2
            output = 10 ** (a + b * x + c * x ** 2)
        else:
            a = 3.287 - 0.401 * z + 0.045 * z ** 2
            b = 1.016 + 0.003 * z + 0.002 * z ** 2
            output = 10 ** (a + b * x)
    return output

def behroozi_2019_raw_with_uncertainties(z, Mhalo, path_to_param_file):
    """
    Stellar mass-halo mass relation from Behroozi +2019.

    This function is a median fit to the raw data for centrals (i.e. excluding
    satellites).
    The stellar mass is the true stellar mass (i.e. w/o observational corrections)
    The fitting function does not include the intrahalo light contribution to the
    stellar mass.
    The halo mass is the peak halo mass that follows the Bryan & Norman (1998)
    spherical overdensity definition
    The function returns the best-fit stellar mass as well as its 68% confidence
    interval range

    The function must be provided with a .txt file containing the fitting parameters
    taken directly from https://www.peterbehroozi.com/data.html

    Parameters
    ----------
    z: float
        Redshift at which SHMH relation is computed
    Mhalo: Any
        Halo mass
    path_to_param_file: str
        Path to the file containing the fitting parameters

    Returns
    -------
    output: Tuple[Any, Any, Any]
        A three-length tuple containing the best-fit stellar mass,
        the 84-percentile of the stellar mass, the 16-percentile of the stellar mass
    """

    Mhalo_log = np.log10(Mhalo)

    # Load the fitting parameters:
    params_list = np.loadtxt(path_to_param_file)

    # Ensure the file has correct dimensions
    assert np.shape(params_list)[1] >= 20

    param_names = (
        "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A "
        "M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z "
        "BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(" ")
    )

    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = np.log(a)

    try:
        log_mstar_all = np.zeros((len(params_list), len(Mhalo)))
    # In case Mhalo does not have __len__ method
    except TypeError:
        log_mstar_all = np.zeros((len(params_list), 1))

    for count, params in enumerate(params_list):

        params = dict(zip(param_names, params))

        zparams = {}

        zparams["m_1"] = (
            params["M_1"]
            + a1 * params["M_1_A"]
            - lna * params["M_1_A2"]
            + z * params["M_1_Z"]
        )
        zparams["sm_0"] = (
            zparams["m_1"]
            + params["EFF_0"]
            + a1 * params["EFF_0_A"]
            - lna * params["EFF_0_A2"]
            + z * params["EFF_0_Z"]
        )
        zparams["alpha"] = (
            params["ALPHA"]
            + a1 * params["ALPHA_A"]
            - lna * params["ALPHA_A2"]
            + z * params["ALPHA_Z"]
        )
        zparams["beta"] = params["BETA"] + a1 * params["BETA_A"] + z * params["BETA_Z"]
        zparams["delta"] = params["DELTA"]
        zparams["gamma"] = 10 ** (
            params["GAMMA"] + a1 * params["GAMMA_A"] + z * params["GAMMA_Z"]
        )

        dm = Mhalo_log - zparams["m_1"]
        dm2 = dm / zparams["delta"]
        logmstar = (
            zparams["sm_0"]
            - np.log10(10 ** (-zparams["alpha"] * dm) + 10 ** (-zparams["beta"] * dm))
            + zparams["gamma"] * np.exp(-0.5 * (dm2 * dm2))
        )

        log_mstar_all[count, :] = logmstar

    # The best-fit stellar mass
    log_mstar_best = np.copy(log_mstar_all[0, :])

    log_mstar_all.sort(axis=0)

    log_mstar_84 = log_mstar_all[int((1 + 0.6827) * len(log_mstar_all) / 2.0)]
    log_mstar_16 = log_mstar_all[int((1 - 0.6827) * len(log_mstar_all) / 2.0)]

    return 10.0 ** log_mstar_best, 10.0 ** log_mstar_84, 10.0 ** log_mstar_16


def plot_halo_evolution(M200, z0, zoutput):

    for M200i in M200:
        M200z = M200accr(M200i, z0, zoutput, cosmo='planck15', commah_flag=True)
        plt.plot(zoutput+1, M200z, '-')


def plot_galaxy_evolution(M200, z0, zoutput):

    num_z_bins = len(zoutput)

    for M200i in M200:
        M200z = M200accr(M200i, z0, zoutput, cosmo='planck15', commah_flag=True)

        M_star = np.zeros(num_z_bins)
        M_84 = np.zeros(num_z_bins)
        M_16 = np.zeros(num_z_bins)

        for i, zi in enumerate(zoutput):

            # Stellar masses (for the given halo masses, at redshift z)
            M_star[i], M_84[i], M_16[i] = behroozi_2019_raw_with_uncertainties(
                zi, M200z[i], "./Behroozi_2019_fitting_params_smhm_true_med_cen.txt"
            )

        y_scatter = np.array((M_star - M_16, M_84 - M_star))

        no_negative = np.where(y_scatter <0)[0]
        y_scatter[no_negative] = 0

        plt.errorbar(zoutput + 1, M_star, yerr=y_scatter, fmt='-', lw=1.5,
                     label='Behroozi et al. (2019)', zorder=0)



if __name__ == '__main__':

    z0 = 0
    M200 = np.arange(9, 13)
    zoutput = np.arange(0, 7, 0.2)


    #################
    # Plot parameters
    params = {
        "font.size": 11,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (6, 2.8),
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.98,
        "figure.subplot.bottom": 0.15,
        "figure.subplot.top": 0.97,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
        "lines.markersize": 2,
        "lines.linewidth": 1.5,
    }
    plt.rcParams.update(params)

    plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_halo_evolution(M200, z0, zoutput)

    plt.axis([1, 5, 1e8, 2e12])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$z$")
    plt.ylabel(r"$M_{200}(z)$ [M$_{\odot}$]")

    xticks = np.array([1, 2, 3, 4, 5])
    labels = ["$0$", "$1$", "$2$", "$3$", "$4$"]
    plt.xticks(xticks, labels)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)

    ###

    ax = plt.subplot(1, 2, 2)
    plt.grid(linestyle='-', linewidth=0.3)

    plot_galaxy_evolution(M200, z0, zoutput)

    plt.axis([1, 5, 1e3, 1e11])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$z$")
    plt.ylabel(r"$M_{*}(z)$ [M$_{\odot}$]")

    xticks = np.array([1, 2, 3, 4, 5])
    labels = ["$0$", "$1$", "$2$", "$3$", "$4$"]
    plt.xticks(xticks, labels)
    ax.tick_params(direction='in', axis='both', which='both', pad=4.5)
    plt.savefig('Example.png', dpi=300)

