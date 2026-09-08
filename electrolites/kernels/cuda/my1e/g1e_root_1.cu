/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "my1e/gint1e_boys.cuh"

__device__
static void GINTfill_int3c1e_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                      const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                      const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double eri = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
        const double eri_per_primitive = GINT_int3c1e_boys0(aij, eij, charge_exponent, omega, r2);
        eri += eri_per_primitive;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    output[i0 + j0 * stride_j + task_grid * stride_ij] = eri;
}

__device__
static void GINTfill_int3c1e_charge_contracted_kernel00(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                        const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                        const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    double eri_grid_sum = 0.0;
    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double Cx = grid_point[0];
        const double Cy = grid_point[1];
        const double Cz = grid_point[2];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double eri_per_grid = 0;
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
            const double aij = a12[ij];
            const double eij = e12[ij];
            const double Px  = x12[ij];
            const double Py  = y12[ij];
            const double Pz  = z12[ij];
            const double PCx = Px - Cx;
            const double PCy = Py - Cy;
            const double PCz = Pz - Cz;
            const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
            const double eri_per_primitive = GINT_int3c1e_boys0(aij, eij, charge_exponent, omega, r2);
            eri_per_grid += eri_per_primitive;
        }

        const double charge = grid_point[3];
        eri_grid_sum += eri_per_grid * charge;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;

    atomicAdd(output + (i0 + j0 * stride_j), eri_grid_sum);
}

__device__
static void GINTfill_int3c1e_density_contracted_kernel00(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                         const BasisProdOffsets offsets, const int nprim_ij,
                                                         const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_grid >= ngrids) {
        return;
    }

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double eri_pair_sum = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {
        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
        // const int* bas_pair2bra = c_bpcache.bas_pair2bra;
        // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
        // const int ish = bas_pair2bra[bas_ij];
        // const int jsh = bas_pair2ket[bas_ij];

        const double* __restrict__ a12 = c_bpcache.a12;
        const double* __restrict__ e12 = c_bpcache.e12;
        const double* __restrict__ x12 = c_bpcache.x12;
        const double* __restrict__ y12 = c_bpcache.y12;
        const double* __restrict__ z12 = c_bpcache.z12;

        double eri_per_pair = 0;
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
            const double aij = a12[ij];
            const double eij = e12[ij];
            const double Px  = x12[ij];
            const double Py  = y12[ij];
            const double Pz  = z12[ij];
            const double PCx = Px - Cx;
            const double PCy = Py - Cy;
            const double PCz = Pz - Cz;
            const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
            const double eri_per_primitive = GINT_int3c1e_boys0(aij, eij, charge_exponent, omega, r2);
            eri_per_pair += eri_per_primitive;
        }

        const double D = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair];
        eri_pair_sum += D * eri_per_pair;
    }
    atomicAdd(output + task_grid, eri_pair_sum);
}

__device__
static void GINTfill_int3c1e_kernel10(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                      const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                      const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double eri_x = 0;
    double eri_y = 0;
    double eri_z = 0;
    for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
        const double aij = a12[ij];
        const double eij = e12[ij];
        const double Px  = x12[ij];
        const double Py  = y12[ij];
        const double Pz  = z12[ij];
        const double PCx = Px - Cx;
        const double PCy = Py - Cy;
        const double PCz = Pz - Cz;
        const double PAx = Px - Ax;
        const double PAy = Py - Ay;
        const double PAz = Pz - Az;
        const double one_over_two_p = 0.5 / aij;
        const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
        double P0, P1;
        GINT_int3c1e_boys01(P0, P1, aij, eij, charge_exponent, omega, r2);
        const double eri_per_primitive_x = P0 * PAx + P1 * PCx * one_over_two_p;
        const double eri_per_primitive_y = P0 * PAy + P1 * PCy * one_over_two_p;
        const double eri_per_primitive_z = P0 * PAz + P1 * PCz * one_over_two_p;
        eri_x += eri_per_primitive_x;
        eri_y += eri_per_primitive_y;
        eri_z += eri_per_primitive_z;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    output[i0 + 0 + j0 * stride_j + task_grid * stride_ij] = eri_x;
    output[i0 + 1 + j0 * stride_j + task_grid * stride_ij] = eri_y;
    output[i0 + 2 + j0 * stride_j + task_grid * stride_ij] = eri_z;
}

__device__
static void GINTfill_int3c1e_charge_contracted_kernel10(double* output, const BasisProdOffsets offsets, const int nprim_ij,
                                                        const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                        const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_ij >= ntasks_ij) {
        return;
    }

    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;

    const int nbas = c_bpcache.nbas;
    const double* __restrict__ bas_x = c_bpcache.bas_coords;
    const double* __restrict__ bas_y = bas_x + nbas;
    const double* __restrict__ bas_z = bas_y + nbas;
    const double Ax = bas_x[ish];
    const double Ay = bas_y[ish];
    const double Az = bas_z[ish];

    double eri_grid_sum_x = 0;
    double eri_grid_sum_y = 0;
    double eri_grid_sum_z = 0;
    for (int task_grid = blockIdx.y * blockDim.y + threadIdx.y; task_grid < ngrids; task_grid += gridDim.y * blockDim.y) {
        const double* grid_point = grid_points + task_grid * 4;
        const double Cx = grid_point[0];
        const double Cy = grid_point[1];
        const double Cz = grid_point[2];
        const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

        double eri_per_grid_x = 0;
        double eri_per_grid_y = 0;
        double eri_per_grid_z = 0;
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
            const double aij = a12[ij];
            const double eij = e12[ij];
            const double Px  = x12[ij];
            const double Py  = y12[ij];
            const double Pz  = z12[ij];
            const double PCx = Px - Cx;
            const double PCy = Py - Cy;
            const double PCz = Pz - Cz;
            const double PAx = Px - Ax;
            const double PAy = Py - Ay;
            const double PAz = Pz - Az;
            const double one_over_two_p = 0.5 / aij;
            const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
            double P0, P1;
            GINT_int3c1e_boys01(P0, P1, aij, eij, charge_exponent, omega, r2);
            const double eri_per_primitive_x = P0 * PAx + P1 * PCx * one_over_two_p;
            const double eri_per_primitive_y = P0 * PAy + P1 * PCy * one_over_two_p;
            const double eri_per_primitive_z = P0 * PAz + P1 * PCz * one_over_two_p;
            eri_per_grid_x += eri_per_primitive_x;
            eri_per_grid_y += eri_per_primitive_y;
            eri_per_grid_z += eri_per_primitive_z;
        }

        const double charge = grid_point[3];
        eri_grid_sum_x += eri_per_grid_x * charge;
        eri_grid_sum_y += eri_per_grid_y * charge;
        eri_grid_sum_z += eri_per_grid_z * charge;
    }

    const int* ao_loc = c_bpcache.ao_loc;
    const int i0 = ao_loc[ish] - ao_offsets_i;
    const int j0 = ao_loc[jsh] - ao_offsets_j;
    atomicAdd(output + (i0 + 0 + j0 * stride_j), eri_grid_sum_x);
    atomicAdd(output + (i0 + 1 + j0 * stride_j), eri_grid_sum_y);
    atomicAdd(output + (i0 + 2 + j0 * stride_j), eri_grid_sum_z);
}

__device__
static void GINTfill_int3c1e_density_contracted_kernel10(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                         const BasisProdOffsets offsets, const int nprim_ij,
                                                         const double omega, const double* grid_points, const double* charge_exponents)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;
    if (task_grid >= ngrids) {
        return;
    }

    const double* grid_point = grid_points + task_grid * 3;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    double eri_pair_sum = 0.0;
    for (int task_ij = blockIdx.x * blockDim.x + threadIdx.x; task_ij < ntasks_ij; task_ij += gridDim.x * blockDim.x) {
        const int bas_ij = offsets.bas_ij + task_ij;
        const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
        const int* bas_pair2bra = c_bpcache.bas_pair2bra;
        // const int* bas_pair2ket = c_bpcache.bas_pair2ket;
        const int ish = bas_pair2bra[bas_ij];
        // const int jsh = bas_pair2ket[bas_ij];

        const double* __restrict__ a12 = c_bpcache.a12;
        const double* __restrict__ e12 = c_bpcache.e12;
        const double* __restrict__ x12 = c_bpcache.x12;
        const double* __restrict__ y12 = c_bpcache.y12;
        const double* __restrict__ z12 = c_bpcache.z12;

        const int nbas = c_bpcache.nbas;
        const double* __restrict__ bas_x = c_bpcache.bas_coords;
        const double* __restrict__ bas_y = bas_x + nbas;
        const double* __restrict__ bas_z = bas_y + nbas;
        const double Ax = bas_x[ish];
        const double Ay = bas_y[ish];
        const double Az = bas_z[ish];

        double eri_per_pair_x = 0;
        double eri_per_pair_y = 0;
        double eri_per_pair_z = 0;
        for (int ij = prim_ij; ij < prim_ij + nprim_ij; ij++) {
            const double aij = a12[ij];
            const double eij = e12[ij];
            const double Px  = x12[ij];
            const double Py  = y12[ij];
            const double Pz  = z12[ij];
            const double PCx = Px - Cx;
            const double PCy = Py - Cy;
            const double PCz = Pz - Cz;
            const double PAx = Px - Ax;
            const double PAy = Py - Ay;
            const double PAz = Pz - Az;
            const double one_over_two_p = 0.5 / aij;
            const double r2 = PCx * PCx + PCy * PCy + PCz * PCz;
            double P0, P1;
            GINT_int3c1e_boys01(P0, P1, aij, eij, charge_exponent, omega, r2);
            const double eri_per_primitive_x = P0 * PAx + P1 * PCx * one_over_two_p;
            const double eri_per_primitive_y = P0 * PAy + P1 * PCy * one_over_two_p;
            const double eri_per_primitive_z = P0 * PAz + P1 * PCz * one_over_two_p;
            eri_per_pair_x += eri_per_primitive_x;
            eri_per_pair_y += eri_per_primitive_y;
            eri_per_pair_z += eri_per_primitive_z;
        }

        // Density element 0 is the 000 element and is not used in McMurchie-Davidson algorithm. Density element 1~3 is the unchanged z,y,x components.
        const double D_x = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 3];
        const double D_y = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 2];
        const double D_z = density[bas_ij - hermite_density_offsets.pair_offset_of_angular_pair + hermite_density_offsets.density_offset_of_angular_pair + hermite_density_offsets.n_pair_of_angular_pair * 1];

        eri_pair_sum += D_x * eri_per_pair_x + D_y * eri_per_pair_y + D_z * eri_per_pair_z;
    }
    atomicAdd(output + task_grid, eri_pair_sum);
}
