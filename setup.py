# coding=utf-8
import setuptools

plugin_identifier = "nozzlealign"
plugin_package = "octoprint_nozzlealign"
plugin_name = "OctoPrint-NozzleAlign"
plugin_version = "0.1.0"
plugin_description = ("Measures the XY offset between two nozzles with a camera on the "
                      "bed, and writes it to the firmware.")
plugin_author = "Chris Nesbitt-Smith"
plugin_author_email = "chris@cns.me.uk"
plugin_url = "https://github.com/chrisns/OctoPrint-NozzleAlign"
plugin_license = "AGPLv3"
plugin_requires = [
    "numpy>=1.21",
    "opencv-python-headless>=4.5",
    "requests",
    "Pillow",
]
plugin_additional_data = []

try:
    import octoprint_setuptools
except ImportError:
    raise SystemExit(
        "Could not import octoprint_setuptools. Install this plugin with the "
        "OctoPrint Python interpreter."
    )

setup_parameters = octoprint_setuptools.create_plugin_setup_parameters(
    identifier=plugin_identifier,
    package=plugin_package,
    name=plugin_name,
    version=plugin_version,
    description=plugin_description,
    author=plugin_author,
    mail=plugin_author_email,
    url=plugin_url,
    license=plugin_license,
    requires=plugin_requires,
    additional_data=plugin_additional_data,
)

setuptools.setup(**setup_parameters)
