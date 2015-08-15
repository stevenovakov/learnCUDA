# Makefile
#     for learnCUDA
#     Copyright (C) 2015 Steve Novakov
#
#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# NOTE: MAKE SURE ALL INDENTS IN MAKEFILE ARE **TABS** AND NOT SPACES
#

CPLR=g++

LIBS= -L /usr/local/cuda/lib64 -lcudart

CPP_FLAGS = -Wall -ansi -pedantic -fPIC -std=c++11
DBG_FLAGS = -g -Wall -ansi -pedantic -fPIC -std=c++11
# This is for a bug with some GCC versions, involving std::thread/pthread where
# they don't get linked properly unless this gets thrown on the end of
# the compile statement
# see:
#  http://stackoverflow.com/questions/17274032/c-threads-stdsystem-error-operation-not-permitted?answertab=votes#tab-top
GCC_PTHREAD_BUG_FLAGS = -pthread -std=c++11

TARGET = program
OBJDIR = lib

CPP_SOURCES := $(wildcard *.cc)
CU_SOURCES := $(wildcard *.cu)
CPP_OBJECTS := $(CPP_SOURCES:%.cc=$(OBJDIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:%.cu=$(OBJDIR)/%.o)

all: $(TARGET)

debug: CPP_FLAGS = $(DBG_FLAGS)
debug: $(TARGET)

$(TARGET): $(OBJDIR) $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CPLR) $(CPP_FLAGS) -o $(TARGET) $(CPP_OBJECTS) $(CU_OBJECTS) $(LIBS) $(GCC_PTHREAD_BUG_FLAGS)

$(CPP_OBJECTS): $(OBJDIR)/%.o:%.cc
	$(CPLR) $(CPP_FLAGS) -c $< -o $@ $(GCC_PTHREAD_BUG_FLAGS)

$(CU_OBJECTS): $(OBJDIR)/%.o:%.cu
	nvcc -arch=sm_30 -c $< -o $@

$(OBJDIR):
	@ mkdir -p $(OBJDIR)

clean:
	$(RM) $(TARGET) $(CPP_OBJECTS) $(CU_OBJECTS) $(OBJDIR)/$(TARGET).so
	$(RM) -rf $(OBJDIR)

