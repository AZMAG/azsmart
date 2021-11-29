"""
Some functions to assist in building data models, workflows and simulations. These
mostly interact with, or extend the functionality of, the task orchestation framework
provided by UDST `orca` and `urbansim_templates` packages.

"""
import gc

import orca
import pandas as pd

from smartpy_core.wrangling import broadcast


@orca.step()
def clean_up():
    """
    Use this at the end of each iterations
    so we don't run out of memory.

    """
    gc.collect()


####################################################
# additional methods for interacting
# w/ orca caches
####################################################


def list_injectable_cache():
    """
    Returns a list of currently cached injectables.

    """
    return [i for i in orca.orca._INJECTABLE_CACHE.keys()]


def list_table_cache():
    """
    Returns a list of currently cached tables.

    """
    return [t for t in orca.orca._TABLE_CACHE.keys()]


def list_column_cache(table_name):
    """
    Returns a list of currently cached columns for the provided table.

    """
    return [c[1] for c in orca.orca._COLUMN_CACHE.keys() if c[0] == table_name]


def is_cached(name):
    """
    Indicates if the item is currently cached.

    Parameters:
    -----------
    name: str or tuple of str
        Name of item to inspect. For columns,
        pass in a tuple/list in the form (table_name, column_name).

    Returns:
    --------
    bool

    """
    if name in list_injectable_cache():
        return True
    if name in list_table_cache():
        return True
    if isinstance(name, tuple) or isinstance(name, list):
        if name[1] in list_column_cache(name[0]):
            return True

    return False


###########################################
# Factory methods for injecting functionss
###########################################


def make_broadcast_injectable(from_table, to_table, col_name, fkey,
                              cache=True, cache_scope='iteration'):
    """
    This creates a broadcast column function/injectable and registers it with orca.

    Parameters:
    -----------
    from_table: str
        The table name to brodacast from (the right).
    to_table: str
        The table name to broadcast to (the left).
    col_name: str
        Name of the column to broadcast.
    fkey: str
        Name of column on the to table that serves as the foreign key.
    cache: bool, optional, default True
        Whether or not the broadcast is cached.
    cache_scope: str, optional, default `iteration`
        Cache scope for the broadcast.

    """
    def broadcast_template():

        return broadcast(
            orca.get_table(from_table)[col_name],
            orca.get_table(to_table)[fkey]
        )

    orca.add_column(to_table, col_name, broadcast_template, cache=cache, cache_scope=cache_scope)


def make_reindex_injectable(from_table, to_table, col_name, cache=True, cache_scope='iteration', fillna=0):
    """
    This creates a PK-PK reindex injectable.

    """
    def reindex_template():
        from_wrap = orca.get_table(from_table)
        to_wrap = orca.get_table(to_table)
        if fillna is None:
            return from_wrap[col_name].reindex(to_wrap.index)
        else:
            return from_wrap[col_name].reindex(to_wrap.index).fillna(fillna)

    orca.add_column(to_table, col_name, reindex_template, cache=cache, cache_scope=cache_scope)


def make_series_broadcast_injectable(from_series, to_table, col_name, fkey, fill_with=None,
                                     cache=True, cache_scope='iteration'):
    """
    Broadcasts an injected series to table.

    """
    def s_broadcast_template():
        b = broadcast(
            orca.get_injectable(from_series),
            orca.get_table(to_table)[fkey]
        )
        if fill_with is not None:
            b.fillna(fill_with, inplace=True)
        return b

    orca.add_column(to_table, col_name, s_broadcast_template, cache=cache, cache_scope=cache_scope)


def register_h5_table(h5, table_name, prefix=None):
    """
    Defines and registers a function for loading a table from
    an h5 file into orca.

    Parameters:
    -----------
    h5: str
        Full path to the h5 containing the table.
    table_name: str
        Table to load, the same name will be used in orca.
    prefix: str, optional default load
        Prefix in the h5 store, eg. `base`, 2010, ...

    """
    full_table_name = table_name
    if prefix is not None:
        full_table_name = '{}/{}'.format(prefix, table_name)

    def load_template():
        with pd.HDFStore(h5, mode='r') as store:
            return store[full_table_name]

    orca.add_table(table_name, load_template, cache=True, cache_scope='forever')
