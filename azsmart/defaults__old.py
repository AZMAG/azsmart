"""
Contains default configuration. To override this just
set orca definitions in the calling notebook or script.


Defines data model for running azsmart simulation
w/ orca and urbansim.

"""

from __future__ import division, print_function

import numpy as np
import pandas as pd
import orca

from azsmart.framework import *
from smartpy_core.wrangling import broadcast


########################
# default settings
########################


# default version/scenario
# this is an idea I'm still working out
orca.add_injectable('sim_version', 'proj1819')


@orca.injectable()
def config_root(sim_version):
    """
    Root directory for configuration files.

    """
    curr_dir = os.path.dirname(__file__)
    return curr_dir.replace(
        'azsmart\\azsmart', 'azsmart\\configs\\{}'.format(sim_version))


# the assumed base year
orca.add_injectable('base_year', 2018)


# the assumed end year
orca.add_injectable('end_year', 2055)


@orca.injectable()
def year(base_year):
    """
    The current year. This will be the base unless called within
    the context of a run.

    """
    if 'iter_var' in orca.list_injectables():
        year = orca.get_injectable('iter_var')
        if year is not None:
            return year

    return base_year


# Adding a couple of lists of MPAs for convenient use
orca.add_injectable(
    'mag_mpas',
    [
        'AJ','FL','GR','MA','PC','QC','AV','BU','CA','CC','CH','EL',
        'FM','FH','GB','GI','GL','GO','GU','LP','CO','ME','PA','PE',
        'PH','SA','SC','SU','TE','TO','WI','YO'
])

orca.add_injectable(
    'cag_mpas',
    [
        'AJ','FL','GR','MA','PC','QC','KE','MM','MR','SP','WK',
        'AK','SN','CG','CL','EY','TC'
    ])

@orca.injectable()
def all_mpas(mag_mpas, cag_mpas):
    mpas = list(set(mag_mpas + cag_mpas))
    mpas.sort()
    return mpas


######################
# default data model
######################


def load_data_model(base_h5=None, prefix='base', tables=None):
    """
    Sets up the data model. Note: this is stil done lazily, this function
    simply defines the functions, the actual loading will be done on demand.

    Parameters:
    -----------
    src_h5: str, optional, default None
        Full path to the h5 file containing base year data.
        If not provided, will pull from orca injectable `base_h5`
    prefix: str, optional, default `base`
        Prefix/directory for the tables in the h5 store.
    tables: list of str, default None
        List of tables to register. If None, all the tables in the h5
        will be registered.

    """
    # source h5 to read tables from
    if base_h5 is None:
        base_h5 = orca.get_injectable('base_h5')

    # list of tables to load
    if tables is None:
        with pd.HDFStore(base_h5, mode='r') as store:
            tables = [t.split('/')[-1] for t in store.keys() if t.startswith('/{}'.format(prefix))]

    # register the tables
    for tab in tables:
        register_h5_table(base_h5, tab, prefix)

    # register columns
    parcel_schema()
    building_schema()
    household_schema()
    seasonal_households_schema()
    person_schema()
    gq_person_schema()
    job_schema()

    # MSA-level controls
    # todo: turn these into a factory?
    @orca.table(cache=True, cache_scope='forever')
    def msa_hh_pop_controls(county_hh_pop_controls):
        """
        Aggregate household pop controls to the msa.

        """
        return county_hh_pop_controls.local.groupby(level=0).sum()


    @orca.table(cache=True, cache_scope='forever')
    def msa_seasonal_controls(county_seasonal_controls):
        """
        Aggregates county seasonal controls to the MSA.

        """
        return county_seasonal_controls.local.groupby(level=0).sum()


    @orca.table(cache=True, cache_scope='forever')
    def msa_gq_controls(county_gq_controls):
        """
        Aggregates county gq controls to the MSA.

        """
        return county_gq_controls.local.groupby(['year', 'gq_type']).sum().reset_index(level=1)


    @orca.table(cache=True, cache_scope='forever')
    def msa_transient_controls(county_transient_controls):
        """
        Aggregates county transient controls to the MSA.

        """
        return county_transient_controls.local.groupby(['year', 'transient_type']).sum()


    @orca.table(cache=True, cache_scope='forever')
    def msa_emp_controls(county_emp_controls):
        """
        Aggregates employment to the MSA.

        """
        return county_emp_controls.local.groupby(
            ['year', 'qcew_naics', 'job_class'])['emp'].sum().reset_index(level=[1, 2])


def parcel_schema():
    """
    Defines/registers parcel columns and broadcast.

    """
    @orca.column('parcels')
    def place_key(parcels, year):
        """
        Year dependent combination of place and county.

        """
        if year == 2017:
            return parcels['county'] + '_' + parcels['city_17']
        else:
            return parcels['county'] + '_' + parcels['city']


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def bldg_sqft(parcels, buildings):
        """
        Total built square feet per parcel.

        """
        b = buildings.to_frame(['parcel_id', 'total_sqft'])
        return b.groupby('parcel_id')['total_sqft'].sum().reindex(parcels.index).fillna(0)


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def bldg_far(parcels, buildings):
        """
        Total build floor-area-ratio for the parcel.
        """
        return fill_nulls(parcels['bldg_sqft'] / parcels['area'])


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def posths_enrollment(parcels, posths):
        """
        Post high school enrollment.

        """
        p = posths.local
        return p.groupby(
            'parcel_id')['enrollment'].sum().reindex(parcels.index).fillna(0)


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def k6_enrollment(parcels, k12):
        """
        Kinder through 6th grade enrollment (elementary).

        """
        k = k12.local
        return k.groupby(
            'parcel_id')['K6'].sum().reindex(parcels.index).fillna(0)


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def g7_8_enrollment(parcels, k12):
        """
        Enrollment for grades 7-8 (middle).

        """
        k = k12.local
        return k.groupby(
            'parcel_id')['G7_8'].sum().reindex(parcels.index).fillna(0)


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def g9_12_enrollment(parcels, k12):
        """
        Enrollment for grades 9-12 (high school).

        """
        k = k12.local
        return k.groupby(
            'parcel_id')['G9_12'].sum().reindex(parcels.index).fillna(0)


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def k12_enrollment(parcels):
        """
        Enrollment for grades kinder through 12th.

        """
        return parcels['k6_enrollment'] + parcels['g7_8_enrollment'] + parcels['g9_12_enrollment']


    @orca.column('parcels', cache=True, cache_scope='iteration')
    def enrollment_all(parcels):
        """
        Enrollment for all grades

        """
        return parcels['k12_enrollment'] + parcels['posths_enrollment']


    @orca.column('parcels', cache=True)
    def is_MC(parcels):
        """
        Dummy for Maricopa County.

        """
        return (parcels.county == 'MC').astype(int)


    @orca.column('parcels')
    def is_PC(parcels):
        """
        Dummy for Pinal County.

        """
        return (parcels.county == 'PC').astype(int)


    @orca.column('parcels', cache=True)
    def is_tribe(parcels):
        """
        Dummy for tribal areas.

        """
        tribal_mpas = ['AK', 'FM', 'GR', 'SA', 'SN', 'TC']
        return parcels['mpa'].isin(tribal_mpas).astype(int)


    @orca.column('parcels', cache=True)
    def east_valley(parcels):
        """
        Dummy for presence in East Valley.

        """
        in_ev = parcels['mpa'].isin([
            'AJ', 'CA', 'CC', 'CH', 'FH', 'FM', 'GC', 'GI',
            'GU', 'ME', 'PA', 'QC', 'SA', 'SC', 'TE'
        ])
        return (parcels['is_MC'] & in_ev).astype(int)


    @orca.column('parcels', cache=True)
    def west_valley(parcels):
        """
        Dummy for presence in West Valley.

        """
        in_wv = parcels['mpa'].isin([
            'AV', 'BU', 'EL', 'GB', 'GL', 'GO', 'LP', 'PE', 'SU', 'TO', 'WI', 'YO'
        ])
        return (parcels['is_MC'] & in_wv).astype(int)


    @orca.column('parcels', cache=True)
    def mpa_sc(parcels):
        """
        Dummy for presence in Scottsdale.

        """
        return (parcels['mpa'] == 'SC').astype(int)


    @orca.column('parcels', cache=True)
    def mpa_ch(parcels):
        """
        Dummy for presence in Chandler.

        """
        return (parcels['mpa'] == 'CH').astype(int)


    @orca.column('parcels', cache=True)
    def mpa_ph(parcels):
        """
        Dummy for presence in Phoenix.

        """
        return (parcels['mpa'] == 'PH').astype(int)


    @orca.column('parcels', cache=True)
    def mpa_pa(parcels):
        """
        Dummy for presence in Paradise Valley.

        """
        return (parcels['mpa'] == 'PA').astype(int)


    @orca.column('parcels')
    def freeway_dist(year, parcels):
        """
        Year dependent freeway distance.

        """
        if year <= 2024:
            return parcels['fwys_2019_dist']
        elif year <= 2030:
            return parcels['fwys_2030_dist']
        else:
            return parcels['fwys_2031_dist']


    # make all parcel columns available
    # Note: this is ugly, but right now I'm hard-coding these so we don't have to
    # load the table first
    parcel_broadcast_cols = [
        'exlu_long_display_id',
        'exlu_display_name',
        'exlu_sector_name',
        'exlu_dev_status',
        'gp_mpa_lucode',
        # 'gp_mag_lucode',
        'dev_objectid',
        'phx_devpolyid',
        'redev_tpcid',
        'city',
        'county',
        'county_fullname',
        'mpa',
        'mpa_fullname',
        'maz',
        'taz',
        'raz',
        'ewc_pinal',
        'city_17',
        'age_restricted',
        'bg_geoid',
        'section_id',
        'hex_id',
        'school_district_name',
        'school_district_id',
        'job_center_id',
        'job_center_name',
        'zcta_geoid',
        'phx_village',
        'phx_aoi',
        'phx_lua_zone',
        'MPO',
        'x',
        'y',
        'area',
        'lr_extensions_dist',
        'fwys_2030_dist',
        'bus_dist',
        'rail_dist',
        'airport_dist',
        'fwys_2024_dist',
        'fwys_2019_dist',
        'cbd_dist',
        'lr_dist',
        'fwys_2031_dist',
        'lr_stop_dist',
        'fwys_2016_dist',
        'place_key',

        'is_MC',
        'is_PC',
        'is_tribe',
        'east_valley',
        'west_valley',
        'mpa_sc',
        'mpa_ch',
        'mpa_ph',
        'mpa_pa',
        'freeway_dist'
    ]

    for par_col in parcel_broadcast_cols:
        for curr_tab in ['buildings', 'households', 'persons', 'jobs',
                         'seasonal_households', 'gq_persons', 'flu_space',
                         'k12', 'posths']:
            make_broadcast_injectable('parcels', curr_tab, par_col, 'parcel_id')


def building_schema():
    """
    Register parcel columns and broadcast.

    """
    @orca.column('buildings', cache=True)
    def is_rsf(buildings):
        """
        Dummy for single family.

        """
        return (buildings['building_type_name'] == 'rsf').astype(int)


    @orca.column('buildings', cache=True)
    def is_rmf(buildings):
        """
        Dummy for multi family.

        """
        return (buildings['building_type_name'] == 'rmf').astype(int)


    @orca.column('buildings', cache=True)
    def is_mf(buildings):
        """
        Duplicate dummy for for multi family. TODO: remove this
        and update variables referencing it

        """
        return buildings['is_rmf']


    @orca.column('buildings', cache=True)
    def is_mh(buildings):
        """
        Dummy for mobile home.

        """
        return (buildings['building_type_name'] == 'mh').astype(int)

    @orca.column('buildings', cache=True)
    def is_med(buildings):
        """
        Dummy for medical buildings.

        """
        return (buildings['building_type_name'] == 'med').astype(int)

    @orca.column('buildings', cache=True)
    def sb_med_sector_sampling_weights(buildings):
        """
        For medical model sector sampling purpose in elcm.

        """
        return (0.75*(buildings['building_type_name'] == 'med') + 0.17*(buildings['building_type_name'] == 'off') \
            + 0.05*(buildings['building_type_name'] == 'retl') + 0.03*(buildings['building_type_name'] == 'gq')).astype(float)

    @orca.column('buildings', 'building_age', cache=True)
    def building_age(buildings, year):
        return year - buildings['year_built']


    @orca.column('buildings', cache=True, cache_scope='forever')
    def sqft_per_res_unit(buildings):
        return fill_nulls(
            buildings['residential_sqft'] / buildings['residential_units'], 0)


    @orca.column('buildings', cache=True, cache_scope='forever')
    def res_sqft_per_unit(buildings):
        """
        TEMPORARY -- this is a duplication of `sqft_per_res_unit'
        so just return it. TODO: eliminate dependencies to this.

        """
        return buildings['sqft_per_res_unit']


    @orca.column('buildings', cache=True, cache_scope='forever')
    def land_area_per_unit(buildings):
        """
        Land area per residential unit, only for residential buildings

        """
        return fill_nulls(buildings['area'] / buildings['residential_units'])


    @orca.column('buildings', cache=True, cache_scope='forever')
    def value_per_res_unit(buildings):
        return fill_nulls(
            buildings['total_fcv'] / buildings['residential_units'], 0)


    @orca.column('buildings', cache=True, cache_scope='forever')
    def value_per_res_sqft(buildings):
        return fill_nulls(
            buildings['total_fcv'] / buildings['residential_sqft'], 0)


    @orca.column('buildings', cache=True, cache_scope='forever')
    def value_per_nonres_sqft(buildings):
        return fill_nulls(
            buildings['total_fcv'] / buildings['non_residential_sqft'], 0)


    @orca.column('buildings', cache=True, cache_scope='forever')
    def is_residential(buildings):
        return buildings['building_type_name'].isin(['rsf', 'rmf', 'mh'])


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def res_hh(buildings, households):
        hh_sums = households.local.groupby('building_id').size()
        return hh_sums.reindex(buildings.index).fillna(0)


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def seas_hh(buildings, seasonal_households):
        seas_sums = seasonal_households.local.groupby('building_id').size()
        return seas_sums.reindex(buildings.index).fillna(0)


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def total_hh(buildings):
        return buildings['res_hh'] + buildings['seas_hh']


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def vac_res_units(buildings):
        return buildings['residential_units'] - buildings['total_hh']


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def site_based_jobs(buildings, jobs):
        sb = jobs.local.query("job_class == 'site based'")
        return sb.groupby('building_id').size().reindex(buildings.index).fillna(0)


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def job_spaces(buildings):
        est_job_spaces = np.round(fill_nulls(
            buildings['non_residential_sqft'] / buildings['sqft_per_job'], 0))
        return pd.DataFrame([est_job_spaces, buildings['site_based_jobs']]).max()


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def vac_job_spaces(buildings):
        return buildings['job_spaces'] - buildings['site_based_jobs']


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def total_sqft(buildings):
        return buildings['residential_sqft'].fillna(0) + buildings['non_residential_sqft'].fillna(0)


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def res_units_in_last5(buildings):
        """
        # of residential units built in the last 5 years.

        """
        is_new = (buildings['building_age'] <= 5).astype(int)
        return buildings['residential_units'] * is_new


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def sf_res_units_in_last5(buildings):
        """
        # of single family residential units built in the last 5 years.

        """
        is_sf = buildings['building_type_name'] == 'rsf'
        is_new = (buildings['building_age'] <= 5)
        return buildings['residential_units'] * (is_sf & is_new).astype(int)


    @orca.column('buildings', cache=True, cache_scope='iteration')
    def mf_res_units_in_last5(buildings):
        """
        # of single family residential units built in the last 5 years.

        """
        is_mf = buildings['building_type_name'] == 'rmf'
        is_new = (buildings['building_age'] <= 5)
        return buildings['residential_units'] * (is_mf & is_new).astype(int)


    @orca.column('buildings')
    def sb_edu_jobs(buildings, jobs):
        """
        Number of naics 61 (education) jobs in a building.

        """
        sb_naics61_jobs = jobs.local.query("job_class == 'site based' and mag_naics == '61'")
        return sb_naics61_jobs.groupby('building_id').size().reindex(buildings.index).fillna(0)


    # broadcast building variables
    bldg_broadcast_cols = ['building_type_name', 'residential_sqft', 'residential_units',
                           'total_fcv', 'year_built', 'parcel_id']

    for bldg_col in bldg_broadcast_cols:
            for curr_tab in ['households', 'persons', 'jobs', 'seasonal_households', 'gq_persons']:
                make_broadcast_injectable('buildings', curr_tab, bldg_col, 'building_id')


def household_schema():
    """
    Register household columns and broadcast.

    """
    @orca.column('households', cache=True, cache_scope='iteration')
    def income_quintile(households):
        """
        Household income quintile. TODO: do we want to do this at the
        county level?

        """
        return pd.Series(
            pd.qcut(
                households['income'], 5, [1, 2, 3, 4, 5]),
            index=households.index
        )


    @orca.column('households', cache=True, cache_scope='iteration')
    def income_quintile_MC(households):
        """
        Household income quintile for MC housedolds only.

        """
        MC_hhlds = households.to_frame(['income', 'county']).query("county == 'MC'")

        return pd.Series(
            pd.qcut(
                MC_hhlds['income'], 5, [1, 2, 3, 4, 5]),
            index=MC_hhlds.index
        )


    @orca.column('households', cache=True, cache_scope='iteration')
    def income_quintile_PC(households):
        """
        Household income quintile for PC households only.

        """
        PC_hhlds = households.to_frame(['income', 'county']).query("county == 'PC'")

        return pd.Series(
            pd.qcut(
                PC_hhlds['income'], 5, [1, 2, 3, 4, 5]),
            index=PC_hhlds.index
        )


    @orca.column('households', cache=True, cache_scope='iteration')
    def income_category(households):
        """
        Define 3 houshold income categories for HLCM submodels

        """
        inc = fill_nulls(households['income'], 0)
        brks = [np.nan, 29999, 99999, np.nan]
        labels = ['income group ' + str(x) for x in range(1, len(brks))]
        with np.errstate(invalid='ignore'):
            # for some reason, the cut method is now throwing a warning about nulls
            # for now, suppress this as the results look fine
            # todo: look into this further
            c = categorize(inc, brks, labels)
        return c


    @orca.column('households', cache=True, cache_scope='iteration')
    def income1(households):
        """
        Dummy for for income group 1

        """
        return (households['income_category'] == 'income group 1').astype(int)


    @orca.column('households', cache=True, cache_scope='iteration')
    def income2(households):
        """
        Dummy for for income group 2

        """
        return (households['income_category'] == 'income group 2').astype(int)


    @orca.column('households', cache=True, cache_scope='iteration')
    def income3(households):
        """
        Dummy for for income group 3

        """
        return (households['income_category'] == 'income group 3').astype(int)


    @orca.column('households', cache=True, cache_scope='iteration')
    def workers(persons, households):
        """
        Number of workers in the household.

        """
        return persons.local.groupby(
            'household_id')['is_worker'].sum().reindex(households.index).fillna(0)


    @orca.column('households', cache=True, cache_scope='iteration')
    def children(persons, households):
        """
        Indicates the presence of children in the household (0, 1).

        """
        min_age = persons.local.groupby(
            'household_id')['age'].min().reindex(households.index).fillna(0)
        return (min_age < 18).astype(int)


    # broadcast household variables
    hh_broadcast_cols = ['building_id', 'income', 'owns', 'time_in_du', 'num_vehicles',
                         'income_quintile', 'year_added']
    for hh_col in hh_broadcast_cols:

            make_broadcast_injectable('households', 'persons', hh_col, 'household_id')


def seasonal_households_schema():
    """
    Seasonal household columns.

    """
    pass


def person_schema():
    """
    Register person columns.

    """
    pass


def gq_person_schema():
    """
    Register person columns.

    """
    pass


def job_schema():
    """
    COlumns and injectables for jobs.

    """

    @orca.injectable(cache=True, cache_scope='forever')
    def job_model_sectors():
        """
        Define a series of naics by job sector.

        Generalizes the 2 digit naics into the sectors we will be
        modeling with. Note: this is being applied to all rows,
        but will only be applicable for site-based jobs.

        """
        return pd.Series({
            '11': 'agriculture',
            '21': 'mining',
            '22': 'utilities',
            '23': 'construction',
            '31': 'manufacturing',
            '32': 'manufacturing',
            '33': 'manufacturing',
            '42': 'warehouse and transport',
            '44': 'retail',
            '45': 'retail',
            '48': 'warehouse and transport',
            '49': 'warehouse and transport',
            '51': 'office',
            '52': 'office',
            '53': 'office',
            '54': 'office',
            '55': 'office',
            '56': 'office',
            '61': 'education',
            '62': 'medical',
            '71': 'hotel arts ent',
            '721': 'hotel arts ent',
            '722': 'retail',
            '81': 'office',
            '92': 'public',
            '93': 'public'
        })


    @orca.injectable(cache=True, cache_scope='forever')
    def qcew_naics_mapping():
        """
        Maps mag naics to the QCEW naics as is used for the controls, e.g. 31-33

        """
        return pd.Series({
            '11': '11',
            '21': '21',
            '22': '22',
            '23': '23',
            '31': '31-33',
            '32': '31-33',
            '33': '31-33',
            '42': '42',
            '44': '44-45',
            '45': '44-45',
            '48': '48-49',
            '49': '48-49',
            '51': '51',
            '52': '52',
            '53': '53',
            '54': '54',
            '55': '55',
            '56': '56',
            '61': '61',
            '62': '62',
            '71': '71',
            '721': '721',
            '722': '722',
            '81': '81',
            '92': '92',
            '93': '93'
        })
