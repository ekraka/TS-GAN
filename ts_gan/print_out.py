#!/usr/bin/env python
def get_header(*args, **kwargs):
    logo_string="""
_____________________________________________________

  *******  *****          *****    *****  *       *
     *    *              *     *  *     * * *     *
     *    *              *        *     * *  *    *
     *     *****   ****  *  ****  ******* *   *   *
     *         *         *     *  *     * *    *  *
     *         *         *     *  *     * *     * *
     *    ******          *****   *     * *       *
____________________________________________________
"""
    print(logo_string)
    header_string="""
    The implementation of generative adversarial 
    networks (GAN) for the prediction of the transition state (TS) 
    geometry based on cartesian coordinates of product and reactant.

    Developed and maintained by Malgorzata-Z Makos and Niraj Verma 

    https://github.com/ekraka/TS-GAN

    Copyright:
    Computational and Theoretical Chemistry Group (CATCO), 
    Department of Chemistry, Southern Methodist University 

    Cite as:
    M.Z. Makoś, N. Verma, E.C. Larson, M. Freindorf, and E. Kraka; 
    J. Chem. Phys. 155, 2021, Vol.155, Issue 2 
    https://doi.org/10.1063/5.0055094

    """

    print(header_string)

def get_end_message(*args, **kwargs):
    end_msg="""
    |-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-|
    |                                 |
    |  TS-GAN completed successfully! |
    |                                 |
    |-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-^-|

    Cite as:
    M.Z. Makoś, N. Verma, E.C. Larson, M. Freindorf, and E. Kraka; 
    J. Chem. Phys. 155, 2021, Vol.155, Issue 2 
    https://doi.org/10.1063/5.0055094

"""
    print(end_msg)





