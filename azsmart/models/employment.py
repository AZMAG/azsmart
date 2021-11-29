"""
Models for simulating employment.

"""

import orca
from urbansim.models.transition import *
from azsmart.utils.dcm2 import *
from azsmart.utils.allocation import AllocationModel, AgentAllocationModel
from smartpy_core.wrangling import broadcast


#########################################################################################
#                                  transition
#########################################################################################


def run_emp_transition(year, controls_wrap, j_wrap, by_county=True):
    """
    Executes the transtion.

    Note:
        - just NEWLY ADDED site-based and wah employment will be un-located.
        - ALL construction and non-sited-based will be un-located.

    """
    controls = controls_wrap.local

    if by_county:
        j = j_wrap.to_frame(j_wrap.local_columns + ['county'])
    else:
        j = j_wrap.local

    # run the transition
    t = TabularTotalsTransition(controls, 'emp')
    tm = TransitionModel(t)
    updated, added, _ = tm.transition(j, year)

    # keep the previous building id
    updated['src_building_id'] = updated['building_id']
    # drop the county? - or do we want it hard-coded??
    if 'county' in updated.columns:
        updated.drop('county', axis=1, inplace=True)

    # flag (un-locate) new jobs
    updated.loc[added, 'year_added'] = year
    updated.loc[added, 'building_id'] = -1
    updated.loc[added, 'magid'] = -1

    # re-shuffle all nsb and constr
    # TODO: would it be helpful to change the job_id as well?
    is_nsb = updated['job_class'].isin(['non site based', 'construction'])
    updated.loc[is_nsb, 'year_added'] = year
    updated.loc[is_nsb, 'building_id'] = -1

    # update the environment
    orca.add_table('jobs', updated)


@orca.step()
def county_emp_transition(year, county_emp_controls, jobs):
    """
    Transition employment using county-level controls.

    """
    run_emp_transition(year, county_emp_controls, jobs, True)


@orca.step()
def msa_emp_transition(year, msa_emp_controls, jobs):
    """
    Transition employment using MSA-level controls.

    """
    run_emp_transition(year, msa_emp_controls, jobs, False)


#########################################################################################
#                             site-based jobs location
#########################################################################################


def clear_sb_jobs_vars():
    """
    Invlidates caches related to site-based jobs.

    """
    orca.clear_columns('jobs')
    orca.clear_columns(
        'buildings',
        [
            'site_based_jobs',
            'vac_job_spaces',
            'sb_edu_jobs'
    ])


# if running a simple jobs re-location, the % to apply
orca.add_injectable('simple_sb_jobs_relocation_pct', .05)


@orca.step()
def simple_sb_jobs_relocation(jobs, simple_sb_jobs_relocation_pct):
    """
    Randomly choose n% of site-based jobs to re-locate.

    For right now ignore public (naics 92).

    """
    sb = jobs.local.query("job_class == 'site based' and mag_naics != '92'")
    num_to_relocate = np.round(len(sb) * simple_sb_jobs_relocation_pct).astype(int)
    ran_idx = np.random.choice(sb.index, num_to_relocate, replace=False)
    jobs.local.loc[ran_idx, 'building_id'] = -1
    clear_sb_jobs_vars()


@orca.injectable(cache=True)
def elcm(config_root):
    """
    Configured location choice model for site-based employment.

    """
    yaml_src = r'{}//Estimation//MCPC//ELCM//elcm_full_segs.yaml'.format(config_root)
    return MnlChoiceModel.from_yaml(str_or_buffer=yaml_src)


@orca.step()
def run_msa_elcm(year, elcm, jobs, buildings):
    """
    Locates site-based jobs. Does not include public (naics 92).

    """

    # get data frames
    # for cols that exist in both, take from buildings
    cols_used = elcm.columns_used()
    bldgs = buildings.to_frame(cols_used)
    job_cols_needed = list(
        set(['building_id', 'job_class']).union(set(cols_used) - set(bldgs.columns)))
    j = jobs.to_frame(job_cols_needed)
    sb = j.query("job_class == 'site based'")

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = elcm.predict(sb, bldgs.dropna(), debug=True)
    choices.dropna(inplace=True)
    choices = choices.astype(j['building_id'].dtype)

    # update the environment
    orca.add_injectable('elcm_samples', samples)
    orca.add_injectable('elcm_choices', choices)
    jobs.local.loc[choices.index, 'building_id'] = choices
    clear_sb_jobs_vars()


@orca.step()
def run_county_elcm(year, elcm, jobs, buildings):
    """
    Locates site-based jobs at the county level. Does not include public (naics 92).

    """

    # get data frames
    # for cols that exist in both, take from buildings

    elcm.predict_sampling_segmentation_col = 'county'
    elcm.predict_sampling_within_percent = 1
    elcm.predict_sampling_within_segments = {'MC': 1, 'PC': 1}
    cols_used = elcm.columns_used()
    bldgs = buildings.to_frame(cols_used + ['county'])
    job_cols_needed = list(
        set(['building_id', 'src_building_id', 'job_class']).union(set(cols_used) - set(bldgs.columns)))
    j = jobs.to_frame(job_cols_needed)
    sb = j.query("job_class == 'site based'")
    sb['county'] = broadcast(buildings['county'], sb['src_building_id'])

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = elcm.predict(sb, bldgs.dropna(), debug=True)
    choices.dropna(inplace=True)
    choices = choices.astype(j['building_id'].dtype)

    # update the environment
    orca.add_injectable('elcm_samples', samples)
    orca.add_injectable('elcm_choices', choices)
    jobs.local.loc[choices.index, 'building_id'] = choices
    clear_sb_jobs_vars()


#########################################################################################
#                                  work at home jobs
#########################################################################################


def clear_wah_jobs_vars():
    """
    Invlidates caches related to work-at-home jobs.

    """
    orca.clear_columns('jobs')


# if running a simple jobs re-location, the % to apply
orca.add_injectable('simple_wah_jobs_relocation_pct', .05)


@orca.step()
def simple_wah_jobs_relocation(jobs, simple_wah_jobs_relocation_pct):
    """
    Randomly choose n% of work at home jobs to re-locate.

    """
    wah = jobs.local.query("job_class == 'work at home'")
    num_to_relocate = np.round(len(wah) * simple_wah_jobs_relocation_pct).astype(int)
    ran_idx = np.random.choice(wah.index, num_to_relocate, replace=False)
    jobs.local.loc[ran_idx, 'building_id'] = -1
    clear_wah_jobs_vars()


@orca.injectable(cache=True)
def wah_lcm(config_root):
    """
    Configured location choice model for work-at-home employment.

    """
    yaml_src = r'{}//Estimation//MCPC//ELCM//wah_full_segs.yaml'.format(config_root)
    return MnlChoiceModel.from_yaml(str_or_buffer=yaml_src)


@orca.step()
def run_msa_wah_lcm(year, wah_lcm, jobs, buildings):
    """
    Locates work at home jobs.

    """

    # get data frames
    # for cols that exist in both, take from buildings
    cols_used = wah_lcm.columns_used()
    bldgs = buildings.to_frame(cols_used)

    job_cols_needed = list(
        set(['building_id', 'job_class']).union(set(cols_used) - set(bldgs.columns)))
    j = jobs.to_frame(job_cols_needed)
    wah = j.query("job_class == 'work at home'")

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = wah_lcm.predict(wah, bldgs.dropna(), debug=True)
    choices = choices['alternative_id']
    choices.dropna(inplace=True)
    choices = choices.astype(j['building_id'].dtype)

    # update the environment
    orca.add_injectable('wah_lcm_samples', samples)
    orca.add_injectable('wah_lcm_choices', choices)
    jobs.local.loc[choices.index, 'building_id'] = choices
    clear_wah_jobs_vars()


@orca.step()
def run_county_wah_lcm(year, wah_lcm, jobs, buildings):
    """
    Locates work at home jobs.

    """

    # get data frames
    # for cols that exist in both, take from buildings

    wah_lcm.predict_sampling_segmentation_col = 'county'
    wah_lcm.predict_sampling_within_percent = 1
    wah_lcm.predict_sampling_within_segments = {'MC': 1, 'PC': 1}
    cols_used = wah_lcm.columns_used()
    bldgs = buildings.to_frame(cols_used)

    job_cols_needed = list(
        set(['building_id', 'src_building_id', 'job_class']).union(set(cols_used) - set(bldgs.columns)))
    j = jobs.to_frame(job_cols_needed)
    wah = j.query("job_class == 'work at home'")
    wah['county'] = broadcast(buildings['county'], wah['src_building_id'])

    # run the prediction
    # note: right now we're dropping nulls
    choices, new_cap, samples = wah_lcm.predict(wah, bldgs.dropna(), debug=True)

    # update the environment
    orca.add_injectable('wah_lcm_samples', samples)
    orca.add_injectable('wah_lcm_choices', choices['alternative_id'])
    jobs.local.loc[choices.index, 'building_id'] = choices['alternative_id'].astype(bldgs.index.dtype)
    clear_wah_jobs_vars()


#----------------------------public  jobs--------------------------------


# define model for public - federal state
@orca.injectable(cache=True)
def pubfs_aa_model():
    PubfsAAModel= AgentAllocationModel(
            allocation_col='pubfs_jobs',
            weight_col='pubfs_jobs')
    return PubfsAAModel


# exectue msa model for public - federal state
@orca.step()
def msa_pubfs_aa_step(iter_var, base_year, pubfs_aa_model, buildings, jobs):
    year = iter_var

    if year > base_year:

        jobs_df = jobs.to_frame(['building_id', 'mag_naics', 'job_class'])
        publicfs_jobs = jobs_df.query("((mag_naics == '93')&(job_class=='site based'))")
        to_locate = publicfs_jobs[publicfs_jobs.building_id==-1]
        num_to_locate = len(to_locate)
        # print "\n # of fed state jobs to locate: \n" + str(num_to_locate)
        if num_to_locate == 0:
            return

        bldgs_df = buildings.to_frame(['pubfs_jobs', 'taz'])
        publicfs_bldgs = bldgs_df.query("pubfs_jobs>0 and taz != 418") # don't more pub jobs go into Luke
        loc_ids, allo = pubfs_aa_model.locate_agents(publicfs_bldgs, to_locate, year)
        jobs_df['building_id'].loc[loc_ids.index] = loc_ids

        #buildings.update_col_from_series('pubfs_jobs', allo)
        jobs.update_col_from_series('building_id', jobs_df.loc[loc_ids.index, 'building_id'], cast=True)
        #jobs.update_col_from_series('building_id', jobs_df.loc[loc_ids.index, 'building_id'].astype(int64))
        # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(int64))
        orca.clear_columns('jobs')
        orca.clear_columns(
            'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'pubfs_jobs'])


# define model for public - local
@orca.injectable(cache=True)
def publ_aa_model():
    PublAAModel= AgentAllocationModel(
            allocation_col='publ_jobs',
            weight_col='publ_weights')
    return PublAAModel


# execute msa model for public - local
@orca.step()
def msa_publ_aa_step(iter_var, base_year, publ_aa_model, buildings, jobs):
    year = iter_var
    if year > base_year:

        jobs_df = jobs.to_frame(['building_id', 'mag_naics', 'job_class'])
        publicl_jobs = jobs_df.query("((mag_naics == '92')&(job_class=='site based'))")
        to_locate = publicl_jobs[publicl_jobs.building_id == -1]
        num_to_locate = len(to_locate)
        # print "\n # of local jobs to locate: \n" + str(num_to_locate)
        if num_to_locate == 0:
            return

        bldgs_df = buildings.to_frame(['publ_weights', 'publ_jobs'])
        public_bldgs = bldgs_df.query("publ_weights>0")
        loc_ids, allo = publ_aa_model.locate_agents(public_bldgs, to_locate, year)
        jobs_df['building_id'].loc[loc_ids.index] = loc_ids

        #buildings.update_col_from_series('publ_jobs', allo)
        jobs.update_col_from_series('building_id', jobs_df.loc[loc_ids.index, 'building_id'], cast=True)
        #jobs.update_col_from_series('building_id', jobs_df.loc[loc_ids.index, 'building_id'].astype(np.int64))
        # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
        orca.clear_columns('jobs')
        orca.clear_columns(
            'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'publ_jobs', 'publ_weights'])

# exectue model for public - federal state by county
@orca.step()
def county_pubfs_aa_step(iter_var, base_year, pubfs_aa_model, buildings, jobs):
    year = iter_var

    if year > base_year:

        # MC
        jobs_df = jobs.to_frame(['building_id', 'mag_naics', 'job_class', 'src_building_id'])
        jobs_df['src_county'] = broadcast(buildings['county'], jobs_df['src_building_id'])
        mc_publicfs_jobs = jobs_df.query("((mag_naics == '93')&(job_class =='site based')&(src_county == 'MC'))")
        mc_to_locate = mc_publicfs_jobs[mc_publicfs_jobs.building_id==-1]
        mc_num_to_locate = len(mc_to_locate)

        if mc_num_to_locate == 0:

            pc_publicfs_jobs = jobs_df.query("((mag_naics == '93')&(job_class =='site based')&(src_county == 'PC'))")
            pc_to_locate = pc_publicfs_jobs[pc_publicfs_jobs.building_id==-1]
            pc_num_to_locate = len(pc_to_locate)

            if pc_num_to_locate == 0:
                return

            bldgs_df = buildings.to_frame(['pubfs_jobs', 'taz', 'county'])
            pc_publicfs_bldgs = bldgs_df.query("pubfs_jobs > 0 and county == 'PC'")
            pc_loc_ids, pc_allo = pubfs_aa_model.locate_agents(pc_publicfs_bldgs, pc_to_locate, year)
            jobs_df['building_id'].loc[pc_loc_ids.index] = pc_loc_ids

            jobs.update_col_from_series('building_id', jobs_df.loc[pc_loc_ids.index, 'building_id'].astype(np.int64))
            # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
            orca.clear_columns('jobs')
            orca.clear_columns(
                'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'pubfs_jobs'])


        else:
            bldgs_df = buildings.to_frame(['pubfs_jobs', 'taz', 'county'])
            mc_publicfs_bldgs = bldgs_df.query("pubfs_jobs > 0 and taz != 418 and county == 'MC'") # don't more pub jobs go into Luke
            mc_loc_ids, mc_allo = pubfs_aa_model.locate_agents(mc_publicfs_bldgs, mc_to_locate, year)
            jobs_df['building_id'].loc[mc_loc_ids.index] = mc_loc_ids

            pc_publicfs_jobs = jobs_df.query("((mag_naics == '93')&(job_class =='site based')&(src_county == 'PC'))")
            pc_to_locate = pc_publicfs_jobs[pc_publicfs_jobs.building_id==-1]
            pc_num_to_locate = len(pc_to_locate)

            if pc_num_to_locate == 0:

                jobs.update_col_from_series('building_id', jobs_df.loc[mc_loc_ids.index, 'building_id'].astype(np.int64))
                # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
                orca.clear_columns('jobs')
                orca.clear_columns(
                    'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'pubfs_jobs'])

                return

            pc_publicfs_bldgs = bldgs_df.query("pubfs_jobs > 0 and county == 'PC'")
            pc_loc_ids, pc_allo = pubfs_aa_model.locate_agents(pc_publicfs_bldgs, pc_to_locate, year)
            jobs_df['building_id'].loc[pc_loc_ids.index] = pc_loc_ids

            jobs.update_col_from_series('building_id', jobs_df.loc[pc_loc_ids.index, 'building_id'].astype(np.int64))
            jobs.update_col_from_series('building_id', jobs_df.loc[mc_loc_ids.index, 'building_id'].astype(np.int64))
            # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
            orca.clear_columns('jobs')
            orca.clear_columns(
                'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'pubfs_jobs'])


# execute model for public - local by county
@orca.step()
def county_publ_aa_step(iter_var, base_year, publ_aa_model, buildings, jobs):
    year = iter_var
    if year > base_year:

        # MC

        jobs_df = jobs.to_frame(['building_id', 'mag_naics', 'job_class', 'src_building_id'])
        jobs_df['src_county'] = broadcast(buildings['county'], jobs_df['src_building_id'])
        mc_publicl_jobs = jobs_df.query("((mag_naics == '92')&(job_class == 'site based')&(src_county == 'MC'))")
        mc_to_locate = mc_publicl_jobs[mc_publicl_jobs.building_id == -1]
        mc_num_to_locate = len(mc_to_locate)

        if mc_num_to_locate == 0:
            pc_publicl_jobs = jobs_df.query("((mag_naics == '92')&(job_class == 'site based')&(src_county == 'PC'))")
            pc_to_locate = pc_publicl_jobs[pc_publicl_jobs.building_id == -1]
            pc_num_to_locate = len(pc_to_locate)

            if pc_num_to_locate == 0:
                return
            bldgs_df = buildings.to_frame(['publ_weights', 'publ_jobs', 'county'])
            pc_public_bldgs = bldgs_df.query("publ_weights>0 and county == 'PC'")
            pc_loc_ids, pc_allo = publ_aa_model.locate_agents(pc_public_bldgs, pc_to_locate, year)
            jobs_df['building_id'].loc[pc_loc_ids.index] = pc_loc_ids


            jobs.update_col_from_series('building_id', jobs_df.loc[pc_loc_ids.index, 'building_id'].astype(np.int64))
            # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
            orca.clear_columns('jobs')
            orca.clear_columns(
                'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'publ_jobs', 'publ_weights'])
        else:
            bldgs_df = buildings.to_frame(['publ_weights', 'publ_jobs', 'county'])
            mc_public_bldgs = bldgs_df.query("publ_weights>0 and county == 'MC'")
            mc_loc_ids, mc_allo = publ_aa_model.locate_agents(mc_public_bldgs, mc_to_locate, year)
            jobs_df['building_id'].loc[mc_loc_ids.index] = mc_loc_ids


            pc_publicl_jobs = jobs_df.query("((mag_naics == '92')&(job_class == 'site based')&(src_county == 'PC'))")
            pc_to_locate = pc_publicl_jobs[pc_publicl_jobs.building_id == -1]
            pc_num_to_locate = len(pc_to_locate)

            if pc_num_to_locate == 0:

                jobs.update_col_from_series('building_id', jobs_df.loc[mc_loc_ids.index, 'building_id'].astype(np.int64))
                # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
                orca.clear_columns('jobs')
                orca.clear_columns(
                    'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'publ_jobs', 'publ_weights'])
                return
            pc_public_bldgs = bldgs_df.query("publ_weights>0 and county == 'PC'")
            pc_loc_ids, pc_allo = publ_aa_model.locate_agents(pc_public_bldgs, pc_to_locate, year)
            jobs_df['building_id'].loc[pc_loc_ids.index] = pc_loc_ids


            jobs.update_col_from_series('building_id', jobs_df.loc[pc_loc_ids.index, 'building_id'].astype(np.int64))
            jobs.update_col_from_series('building_id', jobs_df.loc[mc_loc_ids.index, 'building_id'].astype(np.int64))
            # jobs.update_col_from_series('src_building_id', jobs_df.building_id.astype(np.int64))
            orca.clear_columns('jobs')
            orca.clear_columns(
                'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'publ_jobs', 'publ_weights'])


#########################################################################################
#                                  non site based (nsb) jobs
#########################################################################################


@orca.step()
def non_site_based(year, jobs, buildings):
    """
    Locates non site based jobs to buildings based
    on households and employment
    """

    # get nsb to allocate
    j = jobs.local
    b = buildings.to_frame(['county', 'res_hh', 'site_based_jobs'])
    b['total'] = b.sum(axis=1)

    # retain the county on the new emp and get nsb
    j['county'] = broadcast(buildings['county'], j['src_building_id'])

    for county in ['MC', 'PC']:

        # get current jobs to allocate
        curr_j = j.query(
            "job_class == 'non site based' and building_id == -1 and county == '{}'".format(county))

        # current buildings to allocate to
        curr_b = b.query("total > 0 and county == '{}'".format(county))

        # make the choice
        p = curr_b['total'] / curr_b['total'].sum()

        j.loc[curr_j.index, 'building_id'] = np.random.choice(
            p.index.values, size=len(curr_j), p=p.values, replace=True)

    # tidy up
    # TODO -- see about keeping the county on as a permanent variable.
    j.drop(['county'], axis=1, inplace=True)


@orca.step()
def msa_non_site_based(year, jobs, buildings):
    """
    MSA-level allocation of non-site-based jobs.

    """
    # gather inpputs
    j = jobs.local
    b = buildings.to_frame(['county', 'res_hh', 'site_based_jobs'])
    b['total'] = b.sum(axis=1)

    # make the choice
    curr_j = j.query("job_class == 'non site based' and building_id == -1")
    curr_b = b.query('total > 0')
    p = curr_b['total'] / curr_b['total'].sum()
    choices = np.random.choice(
        p.index.values, size=len(curr_j), p=p.values, replace=True)

    # update the environment
    j.loc[curr_j.index, 'building_id'] = choices


#########################################################################################
#                                  construction jobs
#########################################################################################


@orca.step()
def construction(year, jobs, buildings):
    """
    Locates construction jobs to buildings built in the current
    simulation year
    """

    # get nsb to allocate
    j = jobs.local
    b = buildings.to_frame(['building_id','year_built','residential_sqft','non_residential_sqft','county'])
    b['total'] = b.sum(axis=1)

    # retain the county on the new emp and get nsb
    j['county'] = broadcast(buildings['county'], j['src_building_id'])

    for county in ['MC', 'PC']:

        # get current jobs to allocate
        curr_j = j.query(
            "job_class == 'construction' and building_id == -1 and county == '{}'".format(county))

        # current buildings to allocate to
        curr_b = b.query("total > 0 and county == '{}' and year_built == {}".format(county, year))

        # make the choice
        p = curr_b['total'] / curr_b['total'].sum()

        j.loc[curr_j.index, 'building_id'] = np.random.choice(
            p.index.values, size=len(curr_j), p=p.values, replace=True)

    # tidy up
    # TODO -- see about keeping the county on as a permanent variable.
    j.drop(['county'], axis=1, inplace=True)


@orca.step()
def msa_construction(year, jobs, buildings):
    """
    MSA-level alloction of construction jobs.

    """
    # gather inputs
    j = jobs.local
    b = buildings.to_frame(['building_id', 'year_built', 'residential_sqft', 'non_residential_sqft'])
    b['total'] = b[['residential_sqft', 'non_residential_sqft']].sum(axis=1)

    # make the choice
    curr_j = j.query("job_class == 'construction' and building_id == -1")
    curr_b = b.query('total > 0 and year_built == {}'.format( year))
    p = curr_b['total'] / curr_b['total'].sum()
    choices = np.random.choice(
        p.index.values, size=len(curr_j), p=p.values, replace=True)

    # update the environment
    j.loc[curr_j.index, 'building_id'] = choices


####################
# ad-hoc job updates
####################


@orca.step()
def LAF_pubfs_aa_step(year, base_year, pubfs_aa_model, buildings, jobs, LAF_sb_93_growth):
    """
    Allocates some of the 93 grwoth to Luke Air Force base.

    This should run after transition and before the pulbic job lcm.

    """

    if year in LAF_sb_93_growth.index:

        jobs_df = jobs.to_frame(['building_id', 'mag_naics', 'job_class', 'src_building_id'])
        jobs_df['src_county'] = broadcast(buildings['county'], jobs_df['src_building_id'])
        MC_publicfs_jobs = jobs_df.query("((mag_naics == '93')&(job_class == 'site based')&(src_county == 'MC'))")
        LAF_growth = LAF_sb_93_growth.to_frame().loc[year, 'LAF_SB93_Growth']
        LAF_to_locate_idx = np.random.choice(
            MC_publicfs_jobs[MC_publicfs_jobs.building_id == -1].index, LAF_growth, replace=False)

        bldgs_df = buildings.to_frame(['pubfs_jobs', 'taz'])
        LAF_publicfs_bldgs = bldgs_df.query("pubfs_jobs > 0 and taz == 418")
        LAF_loc_ids, LAF_allo = pubfs_aa_model.locate_agents(
            LAF_publicfs_bldgs, MC_publicfs_jobs.loc[LAF_to_locate_idx], year)
        jobs_df['building_id'].loc[LAF_loc_ids.index] = LAF_loc_ids

        jobs.update_col_from_series('building_id', jobs_df.loc[LAF_loc_ids.index, 'building_id'], cast=True)
        #jobs.update_col_from_series('building_id', jobs_df.loc[LAF_loc_ids.index, 'building_id'].astype(np.int64))

        orca.clear_columns('jobs')
        orca.clear_columns(
            'buildings', ['site_based_jobs', 'job_spaces', 'vac_job_spaces', 'pubfs_jobs'])
