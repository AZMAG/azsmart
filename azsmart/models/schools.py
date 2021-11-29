from __future__ import division

import numpy as np
import pandas as pd
import orca

from smartpy_core.wrangling import *
from smartpy_core.allocation import *


@orca.step()
def calc_school_factors(k12_enrollment_factors,
                        k12_schools_to_sample,
                        posths_enrollment_factors):
    """
    Forces the enrollment factors to get calculated. This
    should only be run in the base year and should be run
    BEFORE transitioning/locating new households.

    """
    pass


###############
# K12
###############


@orca.column('k12', cache=True, cache_scope='iteration')
def total_enrollment(k12):
    """
    Total enrollment across all grade levels.

    """
    return k12['K6'] + k12['G7_8'] + k12['G9_12']


@orca.column('k12', cache=True, cache_scope='iteration')
def remaining_K6_cap(k12):
    """
    Remaining K6 enrollment capacity available at the school.

    """
    return (k12['K6_cap'] - k12['K6']).clip(0)


@orca.column('k12', cache=True, cache_scope='iteration')
def remaining_G7_8_cap(k12):
    """
    Remaining enrollment capacity available for grades 7-8.

    """
    return (k12['G7_8_cap'] - k12['G7_8']).clip(0)


@orca.column('k12', cache=True, cache_scope='iteration')
def remaining_G9_12_cap(k12):
    """
    Remaining enrollment capacity available at the school.

    """
    return (k12['G9_12_cap'] - k12['G9_12']).clip(0)


@orca.table(cache=True, cache_scope='forever')
def k12_enrollment_factors(persons, k12):
    """
    Enrollment rates by age, enrollment group and county. These
    will be used in the schools step to derrive regional
    k12 enrollment.

    """
    pers = persons.to_frame(['age', 'grade_level', 'county'])
    k12 = k12.to_frame(['county', 'K6', 'G7_8', 'G9_12'])

    # assign aggregate school/grade levels
    k12_enroll_groups = pd.Series(
        list(np.repeat('K6', 7)) +
        list(np.repeat('G7_8', 2)) +
        list(np.repeat('G9_12', 4)),
        index=pd.Index(range(0, 13))
    )
    pers['enroll_group'] = broadcast(k12_enroll_groups, pers['grade_level'])

    # sum enrollments from the schools inventory by county and level
    k12_sums = k12.groupby(['county'])[['K6', 'G7_8', 'G9_12']].sum()

    # limit this to 'school-aged' persons
    school_age = pers.query('age > 3 and age < 20')

    # persons enrollment by age
    age_x_group = get_2d_pivot(
        school_age,
        ['county', 'age'],
        'enroll_group'
    )

    # normalize person enrollments to match the inventory
    adj_age_x_group = k12_sums * age_x_group.divide(age_x_group.sum(level=0), axis=1)

    # convert this to shares -- we will use these factors to estimate future enrollments
    return adj_age_x_group.divide(school_age.groupby(['county', 'age']).size(), axis=0)


@orca.table(cache=True, cache_scope='forever')
def k12_schools_to_sample(k12, jobs, buildings, parcels):
    """
    Generates a pool of schools sites to sample from when generating new schools.
    Limit this to school sites that have staff ratios from 6-20.

    This should only be called once.

    """

    # get sites
    sites = k12.to_frame([
        'K6_cap',
        'G7_8_cap',
        'G9_12_cap',
        'staff_ratio',
        'parcel_id',
        'EntityClass',
        'SchoolEntityID'
    ])
    sites.rename(columns={
        'K6_cap': 'K6',
        'G7_8_cap': 'G7_8',
        'G9_12_cap': 'G9_12'
    }, inplace=True)

    # get parcels and buildings sizes
    b = buildings.local
    par = parcels.local
    sites['bldg_sqft'] = broadcast(
        b.groupby('parcel_id')['non_residential_sqft'].sum(), sites['parcel_id']).fillna(0)
    sites['area_sqft'] = broadcast(par['area'], sites['parcel_id'])

    # return school sites that have reasonable enrollment to staff ratios
    return sites.query('staff_ratio >= 6 and staff_ratio <= 20 and bldg_sqft > 0 and area_sqft < 5000000')


@orca.step()
def k12_enroll(year, k12, persons, k12_enrollment_factors, k12_schools_to_sample, flu_space, parcels, buildings):
    """
    Calculates k12 enrollment. Will also add new k12 sites and edu buildings if needed. If developing new sites,
    the projects in flu_space may have their capacities reduced to account for the land utilization.

    Note: right now this is building schools 1 at a time, TODO: look into a more vectorized
    way to do this.

    """

    enroll_cols = ['K6', 'G7_8', 'G9_12']

    # determine if enrollments are constrained by capacity (thus requiring the building of new schools)
    # we might not want to build new schools in the 1st few years?
    constrained = True
    if year < 2021:
        constrained = False

    # schools to sample from
    to_sample = k12_schools_to_sample.local

    # the list of k12 district IDs to ignore -- non-organized areas
    bad_districts = [40, 41, 42, 43, 44, 45]

    # gather tables
    pers = persons.to_frame(['age', 'county', 'raz', 'taz', 'school_district_id', 'year_added'])
    k12_extra_cols = [
        'county', 'taz', 'school_district_id',
        'remaining_K6_cap', 'remaining_G7_8_cap', 'remaining_G9_12_cap',
        'total_enrollment'

    ]
    k12_sites = k12.to_frame(k12.local_columns + k12_extra_cols)
    k12_factors = k12_enrollment_factors.local

    # sum new persons by age and location
    new_pers = pers.query('year_added == {}'.format(year))
    pers_age_sums = new_pers.groupby(
        ['county', 'taz', 'age', 'school_district_id']).size().reset_index(name='persons')

    # estimate enrollment by taz/school district
    factors_b = broadcast(k12_factors, pers_age_sums, ['county', 'age']).fillna(0)
    enroll = factors_b.multiply(pers_age_sums['persons'], axis=0)
    enroll_sums = stochastic_round(
        enroll.groupby([pers_age_sums['taz'], pers_age_sums['school_district_id']]).sum())
    district_sums = enroll_sums.groupby(level=1).sum()
    taz_sums = enroll_sums.groupby(level=0).sum()

    # build new schools if necessary
    if constrained:
        # compare the new enrollees against the available capacity, by district
        cap_sums = k12_sites.groupby(
            'school_district_id')[['remaining_K6_cap', 'remaining_G7_8_cap', 'remaining_G9_12_cap']].sum()
        cap_sums = cap_sums[~cap_sums.index.isin(bad_districts)]
        cap_sums = cap_sums.rename(columns={
            'remaining_K6_cap': 'K6',
            'remaining_G7_8_cap': 'G7_8',
            'remaining_G9_12_cap': 'G9_12',
        })
        district_sums = district_sums.reindex(cap_sums.index).fillna(0)
        shortages = (cap_sums - district_sums)

        # get a taz weight that captures the amount of new enrollment vs. the amount of existing capacity
        taz_cap_sums = k12_sites.groupby(
            'taz')[['K6_cap', 'G7_8_cap', 'G9_12_cap']].sum().reindex(taz_sums.index).fillna(0)
        taz_cap_sums.rename(columns={
            'K6_cap': 'K6',
            'G7_8_cap': 'G7_8',
            'G9_12_cap': 'G9_12',
        }, inplace=True)
        taz_w = fill_nulls(taz_sums / (1 + taz_cap_sums))

        # get projects to put schools on
        # TODO: use something like PLSS zones instead of tazes?
        min_area = 500000  # TODO: do something better here based on the school sampled
        par = parcels.to_frame(['area', 'taz', 'school_district_id'])
        projs = flu_space.to_frame(['building_type', 'parcel_id', 'remaining_land_area', 'total_units', 'land_area'])
        projs['has_edu'] = (projs['building_type'] == 'edu').astype(int)
        par['remaining_land_area'] = projs.groupby('parcel_id')['remaining_land_area'].sum()
        par['has_edu'] = projs.groupby('parcel_id')['has_edu'].sum()
        par = par.query('remaining_land_area >= {}'.format(min_area)).copy()
        for c in enroll_cols:
            par[c] = broadcast(taz_w[c], par['taz'])
            par.dropna(subset=[c], inplace=True)

        # add schools where the district is short
        new_sites = []
        for idx, row in shortages.iterrows():

            # sample schools
            curr_shortages = row.copy()

            while (curr_shortages < 0).any():

                # get the school level with the biggest shortage
                most_short = curr_shortages.idxmin()

                # sample a school site
                curr_sample = to_sample.loc[np.random.choice(
                    to_sample.query('{} > 0'.format(most_short)).index.values, size=1)].copy()
                curr_sample['school_district_id'] = idx

                # update the shortage status
                for c in curr_shortages.index:
                    curr_shortages[c] += curr_sample['{}'.format(c)]

                # locate the site to a parcel
                curr_par = par.query('school_district_id == {}'.format(idx))

                # note: if there are no available parcels
                #       the schools will just grow in place and blow their capacity
                if len(curr_par) > 0:
                    final_w = curr_par[most_short] + curr_par['has_edu']
                    if final_w.sum() > 0:
                        p = (final_w / final_w.sum()).values
                    else:
                        p = None
                    p_choice = np.random.choice(curr_par.index.values, size=1, p=p)
                    curr_sample['parcel_id'] = p_choice
                    curr_sample['parcel_remain_area'] = curr_par.loc[p_choice, 'remaining_land_area'].iloc[0]
                    par.drop(p_choice, inplace=True)
                    new_sites.append(curr_sample)

        # handle new sites
        if len(new_sites) > 0:

            # add schools to the sites
            site_start_id = k12_sites.index.max() + 1
            new_sites = pd.concat(new_sites)
            new_sites.reset_index(drop=True, inplace=True)
            new_sites.index = new_sites.index + site_start_id
            new_sites.index.name = k12_sites.index.name
            for c in enroll_cols:
                new_sites['{}_cap'.format(c)] = new_sites[c]
                new_sites['remaining_{}_cap'.format(c)] = new_sites[c] / 2
                new_sites[c] = 0

            sites_updated = pd.concat([
                k12_sites,
                new_sites
            ])

            # now add the sites as edu buildings
            bldgs = buildings.local
            bldg_start_id = buildings.index.max() + 1
            new_sites.reset_index(drop=True, inplace=True)
            new_sites.index = new_sites.index + bldg_start_id
            new_sites.index.name = 'building_id'
            new_sites['year_built'] = year
            new_sites['building_type_name'] = 'edu'
            new_sites['average_value_per_unit'] = 1
            new_sites['total_fcv'] = 1
            new_sites['sqft_per_job'] = 826
            new_sites['project_id'] = 'k12 schools'
            new_sites['non_residential_sqft'] = new_sites['bldg_sqft']
            new_sites['project_idx'] = np.nan
            for c in ['residential_units', 'residential_sqft',
                      'transient_hh_in_hh', 'transient_hh_in_hotels',
                      'transient_pop_in_hh', 'transient_pop_in_hotels']:
                new_sites[c] = 0

            bldgs_updated = pd.concat([
                bldgs,
                new_sites[bldgs.columns]
            ])

            # remove area/capacity from projects
            chosen_par_areas = new_sites.groupby('parcel_id')['parcel_remain_area'].sum()
            pct_reduce = min_area / chosen_par_areas
            projs_to_reduce = projs[projs['parcel_id'].isin(chosen_par_areas.index)].copy()
            projs_to_reduce['orig_par_area'] = broadcast(chosen_par_areas, projs_to_reduce['parcel_id'])
            projs_to_reduce['pct_reduce'] = broadcast(pct_reduce, projs_to_reduce['parcel_id'])
            projs_to_reduce['new_land_area'] = projs_to_reduce['land_area'] * (1 - projs_to_reduce['pct_reduce'])
            projs_to_reduce['new_units'] = stochastic_round(
                projs_to_reduce['total_units'] * (1 - projs_to_reduce['pct_reduce']))

            # update the environment
            orca.add_table('buildings', bldgs_updated)
            flu_space.local.loc[projs_to_reduce.index, 'units'] = projs_to_reduce['new_units']
            flu_space.local.loc[projs_to_reduce.index, 'land_area'] = projs_to_reduce['new_land_area']

        else:
            # no new sites needed, just use existing
            sites_updated = k12_sites

        # allocate enrollments
        for c in enroll_cols:
            # get the new enrollment
            # keep this column around so we can use it distribute edu job growth
            sites_updated['{}_new'.format(c)] = get_segmented_allocation(
                district_sums[c],
                sites_updated['remaining_{}_cap'.format(c)],
                sites_updated['school_district_id']
            )
            # update the total total enrollment
            sites_updated[c] += sites_updated['{}_new'.format(c)]

        # update the environment
        orca.add_table('k12', sites_updated.drop(k12_extra_cols, axis=1))

    else:
        # unconstrained--just grow in place
        for c in enroll_cols:
            k12_sites['{}_new'.format(c)] = get_segmented_allocation(
                district_sums[c],
                k12_sites[c],
                k12_sites['school_district_id']
            )

            # update the total total enrollment
            k12_sites[c] += k12_sites['{}_new'.format(c)]

        # update the environment
        k12_sites.drop(k12_extra_cols, axis=1, inplace=True)
        orca.add_table('k12', k12_sites)

###############
# posths
###############


@orca.injectable()
def posths_enrollment_online_shares(base_year, year):
    """
    Returns a table of assumed CUMULATIVE CHANGE in online enrollment shares by year.

        - Note we care about just changes, because the base online enrollment in implied by the base
        year inventory and thus already reflected in the posths enrollment factors.

        - For right now, just assume some constant annual changes, extrapolating recent trends seems to
            be too aggressive.

    TODO: do some additional analysis on this, some sources:

        https://www.insidehighered.com/digital-learning/article/2018/11/07/
            new-data-online-enrollments-grow-and-share-overall-enrollment

        https://campustechnology.com/articles/2017/05/02/on-campus-enrollment-shrinks-while-online-continues-its-ascent.aspx

    """

    # for right now, just assume an annual change of .001 for public, 0.003 for private
    s = pd.Series(
        [0.001, 0.001, 0.003, 0.003],
        index=pd.Index([
            'Public 4yr College/University',
            'Public 2yr College',
            'Private University',
            'Trade & Vocational'
        ])
    )

    return s * (year - base_year)


@orca.table(cache=True, cache_scope='forever')
def posths_enrollment_factors(persons, posths):
    """
    Post high school enrollment rates by county, age, income
    and school type (public university, public community college, private university,
    private trade and vocational).

    TODO items:
        - Right now this is done at the county-level does it make more sense to do
            this regionally?

        - Right now this only includes persons in housheholds, might want to include
            GQ in dorms as well?

        - Pinal County only has posths sites for community colleges in the base
                therefore they will only get community college enrollment growth in the future.
                Might need to consider additional growth in the other school types?

        - Do some additional analysis on shares of  online enrollment
            - https://www.insidehighered.com/digital-learning/article/2018/11/07/
                new-data-online-enrollments-grow-and-share-overall-enrollment


    """

    # get enrollment distrbutions from the inventory, by school class/type and county
    posths_df = posths.to_frame(['county', 'gen_class', 'class', 'enrollment'])
    posths_sums = get_2d_pivot(posths_df, ['gen_class', 'class'], 'county', sum_col='enrollment').T

    # get sub-percentages i.e. % of pub in cc vs univ, % of private in univ vs. trades
    #posths_sub_pcts = posths_sums.divide(posths_sums.groupby(level=0, axis=1).sum(axis=0)).fillna(0)
    posths_sub_pcts = posths_sums.divide(posths_sums.groupby(level=0, axis=1).transform('sum'))

    # get student enrollment distribution from persons in households
    pers = orca.get_table('persons').to_frame(
        ['serialno', 'sporder', 'age', 'student_status', 'grade_level', 'county', 'income_quintile'])

    posths_pers_x_income_age = get_2d_pivot(
        pers.query('age > 17 and age <= 65 and grade_level == 13'),
        ['county', 'age', 'income_quintile'],
        'student_status'
    )

    # allocate person enrollments to sub-types
    pers_enroll = posths_pers_x_income_age.multiply(posths_sub_pcts, level=0)

    # control this to the inventories
    adj__pers_enroll = posths_sums * pers_enroll.divide(pers_enroll.sum(level=0), level=0)

    # convert this to age/income quintile shares
    pers_x_income_age = pers.groupby(['county', 'age', 'income_quintile']).size().reindex(adj__pers_enroll.index)

    return adj__pers_enroll.divide(pers_x_income_age, level=1, axis=0).fillna(0)


@orca.step()
def posths_enroll(year, persons, posths,
                  posths_enrollment_factors, posths_enrollment_online_shares):
    """
    Estimates post high school enrollments.

    """
    extra_cols = ['county']
    sites = posths.to_frame(posths.local_columns + extra_cols)
    enroll_factors = posths_enrollment_factors.local
    pers = persons.to_frame(['age', 'county', 'income_quintile', 'year_added'])
    pers = pers.query('year_added == {}'.format(year))

    # get new enrolles by county, school type
    pers_sums = pers.groupby(['county', 'age', 'income_quintile']).size()
    enroll = enroll_factors.multiply(pers_sums, axis=0).reindex(enroll_factors.index)
    enroll = enroll.multiply(1 - posths_enrollment_online_shares, level=1)
    enroll_sums = np.round(enroll.groupby('county').sum().sum(level=1, axis=1))

    # allocate to sites by class, county
    piv = get_2d_pivot(sites.reset_index(), ['posths_id', 'county'], 'class', sum_col='enrollment')
    piv.reset_index(level=1, drop=False, inplace=True)

    for c in enroll_sums.columns:
        piv['{}_new'.format(c)] = get_segmented_allocation(
            enroll_sums[c],
            piv[c],
            piv['county']
        )
    sites['new_enrollment'] = piv[[c for c in piv.columns if c.endswith('_new')]].sum(axis=1)
    sites['enrollment'] += sites['new_enrollment']

    # update the environment
    sites.drop(extra_cols, axis=1, inplace=True)
    orca.add_table('posths', sites)

@orca.step()
def edu_lcm(k12, posths, jobs, buildings):
    """
    County-constrained allocation of edu employment.

    """
    run_edu_lcm(k12, posths, jobs, buildings, True)


@orca.step()
def msa_edu_lcm(k12, posths, jobs, buildings):
    """
    MSA-level (non constrained) allocation of edu
    employment

    """
    run_edu_lcm(k12, posths, jobs, buildings, False)


def run_edu_lcm(k12, posths, jobs, buildings, by_county=True):
    """
    Location choice model for site-based education jobs. Allocates based on new enrollment
    in school sites.

    This may have the addedd side-effect of adding additional education buildings for
    cases where school sites do not have education buildings already present.

    """

    # jobs we need to locate
    j = jobs.to_frame(['job_class', 'mag_naics', 'building_id', 'src_building_id'])
    j = j.query("job_class == 'site based' and mag_naics == '61' and building_id == -1").copy()
    j['county'] = broadcast(buildings['county'], j['src_building_id'])

    # school sites to allocate to
    k12_sites = k12.to_frame(['county', 'K6_new', 'G7_8_new', 'G9_12_new', 'staff_ratio', 'parcel_id'])
    k12_sites['new_enrollment'] = k12_sites[['K6_new', 'G7_8_new', 'G9_12_new']].sum(axis=1)
    k12_sites['type'] = 'k12'

    posths_sites = posths.to_frame(['county', 'new_enrollment', 'staff_ratio', 'parcel_id'])
    posths_sites['type'] = 'post'

    all_sites = pd.concat([
        k12_sites[['county', 'new_enrollment', 'staff_ratio', 'parcel_id', 'type']],
        posths_sites
    ])

    # initial estimate of new employment by school
    all_sites['emp_w'] = all_sites['new_enrollment'] / all_sites['staff_ratio']

    # do we have sites without buildings/edu buildings? -- these are base problems
    # TODO: handle this there!!
    all_sites['edu_bldg_cnt'] = broadcast(
        buildings.local.query("building_type_name == 'edu'").groupby('parcel_id').size(),
        all_sites['parcel_id']
    ).fillna(0)

    # create edu buildings if they're missing
    need_bldgs = all_sites.query('edu_bldg_cnt == 0').copy()
    if len(need_bldgs) > 0:
        need_bldgs['building_type_name'] = 'edu'
        need_bldgs['year_built'] = 1979
        need_bldgs['average_value_per_unit'] = 1
        need_bldgs['total_fcv'] = 1
        need_bldgs['sqft_per_job'] = 826
        need_bldgs['project_id'] = 'missing edu space'
        need_bldgs['non_residential_sqft'] = 1
        need_bldgs['project_idx'] = np.nan
        for c in ['residential_units', 'residential_sqft',
                  'transient_hh_in_hh', 'transient_hh_in_hotels',
                  'transient_pop_in_hh', 'transient_pop_in_hotels']:
            need_bldgs[c] = 0

        exist_b = buildings.local
        start_id = exist_b.index.max() + 1
        need_bldgs.reset_index(inplace=True)
        need_bldgs.index = need_bldgs.index.values + start_id
        need_bldgs.index.name = 'building_id'

        bldgs = pd.concat([
            exist_b,
            need_bldgs[exist_b.columns]
        ])
        orca.add_table('buildings', bldgs)

        all_sites['edu_bldg_cnt'].replace(0, 1, inplace=True)
    else:
        bldgs = buildings.local

    # get spaces to allocate to
    m = pd.merge(
        all_sites,
        bldgs.query("building_type_name == 'edu'")[['parcel_id']].reset_index(),
        on='parcel_id'
    )
    m['final_w'] = m['emp_w'] / m['edu_bldg_cnt']

    # assign jobs to sites
    def to_sites(curr_j, curr_spaces):
        p = curr_spaces['final_w'] / curr_spaces['final_w'].sum()
        choices = np.random.choice(curr_spaces['building_id'].values, size=len(curr_j), p=p)
        j.loc[curr_j.index, 'building_id'] = choices

    if by_county:
        for c in ['MC', 'PC']:
            curr_j = j.query("county == '{}'".format(c))
            curr_spaces = m.query("county == '{}'".format(c))
            to_sites(curr_j, curr_spaces)
    else:
        to_sites(j, m)

    #for c in ['MC', 'PC']:
    #    curr_j = j.query("county == '{}'".format(c))
    #    curr_spaces = m.query("county == '{}'".format(c))
    #    p = curr_spaces['final_w'] / curr_spaces['final_w'].sum()
    #    choices = np.random.choice(curr_spaces['building_id'].values, size=len(curr_j), p=p)
    #    j.loc[curr_j.index, 'building_id'] = choices

    jobs.local.loc[j.index, 'building_id'] = j['building_id']
    orca.clear_columns('jobs')
    orca.clear_columns('buildings', ['site_based_jobs', 'vac_job_spaces', 'sb_edu_jobs'])
