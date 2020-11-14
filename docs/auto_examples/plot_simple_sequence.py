
"""
Simple Sequence Plot
===========================

This is a very simple example of plotting a sequence using the :meth:`plot_sequence() <pysan.core.plot_sequence>` method.
"""

import pysan as ps

sequence = [1,1,1,2,2,3,2,2,3,3,2,1,1,2,3,3,3,2,2,2,3,2,1,1]

ps.plot_sequence(sequence)

###############################################################################
#
#
# The output can be saved as a variable if you'd like to save or customise the plot further.
# Remember that all pysan plots return `matplotlib plot <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html>`_ objects, so can be adjusted as all plt objects can!
# For a quick guide on customisation using matplotlib I recommend `this earthdatascience post <https://www.earthdatascience.org/courses/scientists-guide-to-plotting-data-in-python/plot-with-matplotlib/introduction-to-matplotlib-plots/customize-plot-colors-labels-matplotlib/>`_.


plot = ps.plot_sequence(sequence)

plot.grid(True)

plot.savefig('simple_sequence.png')