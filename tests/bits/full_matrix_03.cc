//----------------------------------------------------------------------
//    $Id$
//    Version: $Name$ 
//
//    Copyright (C) 2007 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------------------------------------------------


// check running over const iterators starting at the second line


#include "../tests.h"
#include "full_matrix_common.h"


std::string output_file_name = "full_matrix_03/output";


template <typename number>
void
check ()
{
  FullMatrix<number> m;
  make_matrix (m);

  
  for (typename FullMatrix<number>::const_iterator
	 p = m.begin(1); p!=m.end(1); ++p)
    deallog << p->row() << ' ' << p->column() << ' '
	    << p->value()
	    << std::endl;
}

