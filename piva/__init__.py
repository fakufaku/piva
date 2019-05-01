#  Package initialization and imports.
#  Copyright (C) 2020  Robin Scheibler
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

__version__ = "0.0.1a"

from . import metrics

# import the core functions
from .auxiva import auxiva, overiva
from .auxiva_iss import auxiva_iss
from .five import five
from .pca import pca
from .projection_back import project_back
from .utils import crandn, tensor_H
from .defaults import models

from .core import stft, istft

algorithms = {
    "auxiva": auxiva,
    "auxiva_iss": auxiva_iss,
    "five": five,
    "overiva": overiva,
}
