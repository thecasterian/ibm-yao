#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>
#include "HYPRE_struct_ls.h"
#include "_hypre_struct_mv.h"

#define Nx 100
#define Ny 100

#define rho 1.
#define nu 1e-3

#define dt 0.01
#define numtsteps 5000

double xf[Nx+1], yf[Ny+1];
double xc[Nx+2], yc[Ny+2], dx[Nx+2], dy[Ny+2];
double aw[Nx+1], ae[Nx+1], as[Ny+2], an[Ny+2];
double bw[Nx+2], be[Nx+2], bs[Ny+1], bn[Ny+1];
double cw[Nx+2], ce[Nx+2], cs[Ny+2], cn[Ny+2], csum[Nx+2][Ny+2];

double u[Nx+1][Ny+2], u_star[Nx+1][Ny+2];
double v[Nx+2][Ny+1], v_star[Nx+2][Ny+1];
double p[Nx+2][Ny+2], p_prime[Nx+2][Ny+2];

double N1[Nx+1][Ny+2], N1_prev[Nx+1][Ny+2];
double N2[Nx+2][Ny+1], N2_prev[Nx+2][Ny+1];

HYPRE_StructGrid grid_u, grid_v, grid_p;
HYPRE_StructStencil stencil_u, stencil_v, stencil_p;
HYPRE_StructMatrix A_u, A_v, A_p;
HYPRE_StructVector b_u, b_v, b_p;
HYPRE_StructVector x_u, x_v, x_p;
HYPRE_StructSolver solver_u, solver_v, solver_p;
HYPRE_StructSolver precond;

double RHS_u[(Nx-1)*Ny], RHS_v[Nx*(Ny-1)];
double RHS_p[Nx*Ny];

void calc_mesh(void);
void build_hypre(void);
void init(void);
void calc_N(void);
void calc_inter_vel(void);
void calc_vel_RHS(void);
void update_inter_vel(void);
void calc_pre_corr(void);
void calc_pre_RHS(void);
void update_pre_corr(void);
void update_all(void);
void export(void);

int main(int argc, char *argv[]) {
    int nprocs, rank;

    MPI_Init(&argc, &argv);
    HYPRE_Init();

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (nprocs > 1 && rank == 0) {
        printf("error: nprocs != 1\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (int i = 0; i <= Nx; i++)
        xf[i] = (double)i / Nx;
    for (int j = 0; j <= Ny; j++)
        yf[j] = (double)j / Ny;
    calc_mesh();

    build_hypre();

    init();

    calc_N();
    memcpy(N1_prev, N1, sizeof(N1));
    memcpy(N2_prev, N2, sizeof(N2));

    for (int tstep = 1; tstep <= numtsteps; tstep++) {
        calc_N();
        calc_inter_vel();
        calc_pre_corr();
        update_all();
        if (tstep % 100 == 0) printf("%d\n", tstep);
    }

    export();

    HYPRE_Finalize();
    MPI_Finalize();

    return 0;
}

void calc_mesh(void) {
    for (int i = 1; i <= Nx; i++) xc[i] = (xf[i-1] + xf[i]) / 2;
    for (int j = 1; j <= Ny; j++) yc[j] = (yf[j-1] + yf[j]) / 2;
    xc[0] = 2*xf[0] - xc[1];
    xc[Nx+1] = 2*xf[Nx] - xc[Nx];
    yc[0] = 2*yf[0] - yc[1];
    yc[Ny+1] = 2*yf[Ny] - yc[Ny];

    for (int i = 1; i <= Nx; i++) dx[i] = xf[i] - xf[i-1];
    for (int j = 1; j <= Ny; j++) dy[j] = yf[j] - yf[j-1];
    dx[0] = dx[1];
    dx[Nx+1] = dx[Nx];
    dy[0] = dy[1];
    dy[Ny+1] = dy[Ny];

    for (int i = 0; i <= Nx; i++) {
        aw[i] = 1/((xc[i+1]-xc[i])*dx[i]);
        ae[i] = 1/((xc[i+1]-xc[i])*dx[i+1]);
    }
    for (int j = 1; j <= Ny; j++) {
        as[j] = 1/((yc[j]-yc[j-1])*dy[j]);
        an[j] = 1/((yc[j+1]-yc[j])*dy[j]);
    }

    for (int i = 1; i <= Nx; i++) {
        bw[i] = 1/((xc[i]-xc[i-1])*dx[i]);
        be[i] = 1/((xc[i+1]-xc[i])*dx[i]);
    }
    for (int j = 0; j <= Ny; j++) {
        bs[j] = 1/((yc[j+1]-yc[j])*dy[j]);
        bn[j] = 1/((yc[j+1]-yc[j])*dy[j+1]);
    }

    for (int i = 1; i <= Nx; i++) {
        cw[i] = 1/((xc[i]-xc[i-1])*dx[i]);
        ce[i] = 1/((xc[i+1]-xc[i])*dx[i]);
    }
    for (int j = 1; j <= Ny; j++) {
        cs[j] = 1/((yc[j]-yc[j-1])*dy[j]);
        cn[j] = 1/((yc[j+1]-yc[j])*dy[j]);
    }
    for (int i = 1; i <= Nx; i++)
        for (int j = 1; j <= Ny; j++)
            csum[i][j] = cw[i] + ce[i] + cs[j] + cn[j];
}

void build_hypre(void) {
    /* u. */
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx-1, Ny};
        HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid_u);
        HYPRE_StructGridSetExtents(grid_u, ilower, iupper);
        HYPRE_StructGridAssemble(grid_u);
    }
    {
        /* center, west, east, south, north. */
        HYPRE_Int offset[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        HYPRE_StructStencilCreate(2, 5, &stencil_u);
        for (int i = 0; i < 5; i++)
            HYPRE_StructStencilSetElement(stencil_u, i, offset[i]);
    }
    {
        HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid_u, stencil_u, &A_u);
        HYPRE_StructMatrixInitialize(A_u);

        for (int i = 1; i <= Nx-1; i++)
            for (int j = 1; j <= Ny; j++) {
                HYPRE_Int ii[2] = {i, j};
                HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
                double values[5] = {
                    1 + nu*dt/2*(aw[i]+ae[i]+as[j]+an[j]),
                    -nu*dt/2*aw[i],
                    -nu*dt/2*ae[i],
                    -nu*dt/2*as[j],
                    -nu*dt/2*an[j]
                };
                if (i == 1) values[1] = 0;
                if (i == Nx-1) values[2] = 0;
                if (j == 1) {
                    values[0] -= values[3];
                    values[3] = 0;
                }
                if (j == Ny) {
                    values[0] -= values[4];
                    values[4] = 0;
                }
                HYPRE_StructMatrixSetBoxValues(A_u, ii, ii,
                                               5, stencil_indices, values);
            }

        HYPRE_StructMatrixAssemble(A_u);
        HYPRE_StructMatrixPrint("A_u", A_u, 0);
    }
    {
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_u, &b_u);
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_u, &x_u);
        HYPRE_StructVectorInitialize(b_u);
        HYPRE_StructVectorInitialize(x_u);
    }

    /* v. */
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny-1};
        HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid_v);
        HYPRE_StructGridSetExtents(grid_v, ilower, iupper);
        HYPRE_StructGridAssemble(grid_v);
    }
    {
        /* center, west, east, south, north. */
        HYPRE_Int offset[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        HYPRE_StructStencilCreate(2, 5, &stencil_v);
        for (int i = 0; i < 5; i++)
            HYPRE_StructStencilSetElement(stencil_v, i, offset[i]);
    }
    {
        HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid_v, stencil_v, &A_v);
        HYPRE_StructMatrixInitialize(A_v);

        for (int i = 1; i <= Nx; i++)
            for (int j = 1; j <= Ny-1; j++) {
                HYPRE_Int ii[2] = {i, j};
                HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
                double values[5] = {
                    1 + nu*dt/2*(bw[i]+be[i]+bs[j]+bn[j]),
                    -nu*dt/2*bw[i],
                    -nu*dt/2*be[i],
                    -nu*dt/2*bs[j],
                    -nu*dt/2*bn[j]
                };
                if (i == 1) {
                    values[0] -= values[1];
                    values[1] = 0;
                }
                if (i == Nx) {
                    values[0] -= values[2];
                    values[2] = 0;
                }
                if (j == 1) values[3] = 0;
                if (j == Ny-1) values[4] = 0;
                HYPRE_StructMatrixSetBoxValues(A_v, ii, ii,
                                               5, stencil_indices, values);
            }

        HYPRE_StructMatrixAssemble(A_v);
        HYPRE_StructMatrixPrint("A_v", A_v, 0);
    }
    {
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_v, &b_v);
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_v, &x_v);
        HYPRE_StructVectorInitialize(b_v);
        HYPRE_StructVectorInitialize(x_v);
    }

    /* p. */
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny};
        HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid_p);
        HYPRE_StructGridSetExtents(grid_p, ilower, iupper);
        HYPRE_StructGridAssemble(grid_p);
    }
    {
        /* center, west, east, south, north. */
        HYPRE_Int offset[5][2] = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        HYPRE_StructStencilCreate(2, 5, &stencil_p);
        for (int i = 0; i < 5; i++)
            HYPRE_StructStencilSetElement(stencil_p, i, offset[i]);
    }
    {
        HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid_p, stencil_p, &A_p);
        HYPRE_StructMatrixInitialize(A_p);

        for (int i = 1; i <= Nx; i++)
            for (int j = 1; j <= Ny; j++) {
                HYPRE_Int ii[2] = {i, j};
                HYPRE_Int stencil_indices[5] = {0, 1, 2, 3, 4};
                double values[5] = {
                    1,
                    -cw[i]/csum[i][j],
                    -ce[i]/csum[i][j],
                    -cs[j]/csum[i][j],
                    -cn[j]/csum[i][j]
                };
                if (i == 1) {
                    values[0] += values[1];
                    values[1] = 0;
                }
                if (i == Nx) {
                    values[0] += values[2];
                    values[2] = 0;
                }
                if (j == 1) {
                    values[0] += values[3];
                    values[3] = 0;
                }
                if (j == Ny) {
                    values[0] += values[4];
                    values[4] = 0;
                }
                if (i == 1 && j == 1) {
                    values[0] = 1;
                    values[1] = values[2] = values[3] = values[4] = 0;
                }
                HYPRE_StructMatrixSetBoxValues(A_p, ii, ii,
                                               5, stencil_indices, values);
            }

        HYPRE_StructMatrixAssemble(A_p);
        HYPRE_StructMatrixPrint("A_p", A_p, 0);
    }
    {
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_p, &b_p);
        HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid_p, &x_p);
        HYPRE_StructVectorInitialize(b_p);
        HYPRE_StructVectorInitialize(x_p);
    }

    /* Solvers. */
    {
        HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver_u);
        HYPRE_StructBiCGSTABSetTol(solver_u, 1e-6);
        HYPRE_StructBiCGSTABSetMaxIter(solver_u, 1000);
        // HYPRE_StructBiCGSTABSetPrintLevel(solver_u, 2);

        HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver_v);
        HYPRE_StructBiCGSTABSetTol(solver_v, 1e-6);
        HYPRE_StructBiCGSTABSetMaxIter(solver_v, 1000);
        // HYPRE_StructBiCGSTABSetPrintLevel(solver_v, 2);

        HYPRE_StructBiCGSTABCreate(MPI_COMM_WORLD, &solver_p);
        HYPRE_StructBiCGSTABSetTol(solver_p, 1e-6);
        HYPRE_StructBiCGSTABSetMaxIter(solver_p, 1000);
        // HYPRE_StructBiCGSTABSetPrintLevel(solver_p, 2);
    }
}

void init(void) {
    memset(u, 0, sizeof(u));
    memset(v, 0, sizeof(v));
    memset(p, 0, sizeof(p));
    for (int i = 0; i <= Nx; i++) u[i][Ny+1] = 2;
    memcpy(u_star, u, sizeof(u));
}

void calc_N(void) {
    for (int i = 1; i <= Nx-1; i++)
        for (int j = 1; j <= Ny; j++)
            N1[i][j]
                = 1/(dx[i]+dx[i+1]) * ((u[i-1][j]+u[i][j])/2*(u[i][j]-u[i-1][j])/dx[i]*dx[i+1]
                                       + (u[i][j]+u[i+1][j])/2*(u[i+1][j]-u[i][j])/dx[i+1]*dx[i])
                + .5 * ((v[i][j-1]*dx[i+1]+v[i+1][j-1]*dx[i])/(dx[i]+dx[i+1])*(u[i][j]-u[i][j-1])/(yc[j]-yc[j-1])
                        + (v[i][j]*dx[i+1]+v[i+1][j]*dx[i])/(dx[i]+dx[i+1])*(u[i][j+1]-u[i][j])/(yc[j+1]-yc[j]));
    for (int i = 1; i <= Nx; i++)
        for (int j = 1; j <= Ny-1; j++)
            N2[i][j]
                = .5 * ((u[i-1][j]*dy[j+1]+u[i-1][j+1]*dy[j])/(dy[j]+dy[j+1])*(v[i][j]-v[i-1][j])/(xc[i]-xc[i-1])
                        + (u[i][j]*dy[j+1]+u[i][j+1]*dy[j])/(dy[j]+dy[j+1])*(v[i+1][j]-v[i][j])/(xc[i+1]-xc[i]))
                + 1/(dy[j]+dy[j+1]) * ((v[i][j-1]+v[i][j])/2*(v[i][j]-v[i][j-1])/dy[j]*dy[j+1]
                                       + (v[i][j]+v[i][j+1])/2*(v[i][j+1]-v[i][j])/dy[j+1]*dy[j]);
}

void calc_inter_vel(void) {
    calc_vel_RHS();
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx-1, Ny};
        int m;

        HYPRE_StructVectorSetBoxValues(b_u, ilower, iupper, RHS_u);
        HYPRE_StructVectorAssemble(b_u);

        memset(RHS_u, 0, sizeof(RHS_u));
        HYPRE_StructVectorSetBoxValues(x_u, ilower, iupper, RHS_u);
        HYPRE_StructVectorAssemble(x_u);

        HYPRE_StructBiCGSTABSetup(solver_u, A_u, b_u, x_u);
        HYPRE_StructBiCGSTABSolve(solver_u, A_u, b_u, x_u);

        HYPRE_StructVectorGetBoxValues(x_u, ilower, iupper, RHS_u);
        m = 0;
        for (int j = 1; j <= Ny; j++)
            for (int i = 1; i <= Nx-1; i++)
                u_star[i][j] = RHS_u[m++];
    }
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny-1};
        int m;

        HYPRE_StructVectorSetBoxValues(b_v, ilower, iupper, RHS_v);
        HYPRE_StructVectorAssemble(b_v);

        memset(RHS_v, 0, sizeof(RHS_v));
        HYPRE_StructVectorSetBoxValues(x_v, ilower, iupper, RHS_v);
        HYPRE_StructVectorAssemble(x_v);

        HYPRE_StructBiCGSTABSetup(solver_v, A_v, b_v, x_v);
        HYPRE_StructBiCGSTABSolve(solver_v, A_v, b_v, x_v);

        HYPRE_StructVectorGetBoxValues(x_v, ilower, iupper, RHS_v);
        m = 0;
        for (int j = 1; j <= Ny-1; j++)
            for (int i = 1; i <= Nx; i++)
                v_star[i][j] = RHS_v[m++];
    }
    update_inter_vel();
}

void calc_vel_RHS(void) {
    int m;

    m = 0;
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx-1; i++) {
            RHS_u[m]
                = -dt/2*(3*N1[i][j]-N1_prev[i][j]) - dt/rho*(p[i+1][j]-p[i][j])/(xc[i+1]-xc[i])
                + u[i][j] + nu*dt/2 * (aw[i]*u[i-1][j] + ae[i]*u[i+1][j] + as[j]*u[i][j-1] + an[j]*u[i][j+1]
                                       - (aw[i]+ae[i]+as[j]+an[j])*u[i][j]);
            if (j == Ny) RHS_u[m] += nu*dt*an[j];
            m++;
        }

    m = 0;
    for (int j = 1; j <= Ny-1; j++)
        for (int i = 1; i <= Nx; i++) {
            RHS_v[m]
                = -dt/2*(3*N2[i][j]-N2_prev[i][j]) - dt/rho*(p[i][j+1]-p[i][j])/(yc[j+1]-yc[j])
                + v[i][j] + nu*dt/2 * (bw[i]*v[i-1][j] + be[i]*v[i+1][j] + bs[j]*v[i][j-1] + bn[j]*v[i][j+1]
                                       - (bw[i]+be[i]+bs[j]+bn[j])*v[i][j]);
            m++;
        }
}

void update_inter_vel(void) {
    for (int j = 0; j <= Ny+1; j++) u_star[0][j] = u_star[Nx][j] = 0;
    for (int i = 0; i <= Nx; i++) {
        u_star[i][0] = -u_star[i][1];
        u_star[i][Ny+1] = 2-u_star[i][Ny];
    }

    for (int i = 0; i <= Nx+1; i++) v_star[i][0] = v_star[i][Ny] = 0;
    for (int j = 0; j <= Ny; j++) {
        v_star[0][j] = -v_star[1][j];
        v_star[Nx+1][j] = -v_star[Nx][j];
    }
}

void calc_pre_corr(void) {
    calc_pre_RHS();
    {
        HYPRE_Int ilower[2] = {1, 1}, iupper[2] = {Nx, Ny};
        int m;

        HYPRE_StructVectorSetBoxValues(b_p, ilower, iupper, RHS_p);
        HYPRE_StructVectorAssemble(b_p);

        memset(RHS_p, 0, sizeof(RHS_p));
        HYPRE_StructVectorSetBoxValues(x_p, ilower, iupper, RHS_p);
        HYPRE_StructVectorAssemble(x_p);

        HYPRE_StructBiCGSTABSetup(solver_p, A_p, b_p, x_p);
        HYPRE_StructBiCGSTABSolve(solver_p, A_p, b_p, x_p);

        HYPRE_StructVectorGetBoxValues(x_p, ilower, iupper, RHS_p);
        m = 0;
        for (int j = 1; j <= Ny; j++)
            for (int i = 1; i <= Nx; i++)
                p_prime[i][j] = RHS_p[m++];
    }
    update_pre_corr();
}

void calc_pre_RHS(void) {
    int m;

    m = 0;
    for (int j = 1; j <= Ny; j++)
        for (int i = 1; i <= Nx; i++) {
            RHS_p[m] = -rho/(dt*csum[i][j]) * ((u_star[i][j]-u_star[i-1][j])/dx[i]
                                               + (v_star[i][j]-v_star[i][j-1])/dy[j]);
            if (i == 1 && j == 1) RHS_p[m] = 0;
            m++;
        }
}

void update_pre_corr(void) {
    for (int i = 0; i <= Nx+1; i++) {
        p_prime[i][0] = p_prime[i][1];
        p_prime[i][Ny+1] = p_prime[i][Ny];
    }
    for (int j = 0; j <= Ny+1; j++) {
        p_prime[0][j] = p_prime[1][j];
        p_prime[Nx+1][j] = p_prime[Nx][j];
    }
}

void update_all(void) {
    for (int i = 1; i <= Nx-1; i++)
        for (int j = 1; j <= Ny; j++)
            u[i][j] = u_star[i][j] - dt/rho * (p_prime[i+1][j]-p_prime[i][j])/(xc[i+1]-xc[i]);
    for (int i = 1; i <= Nx; i++)
        for (int j = 1; j <= Ny-1; j++)
            v[i][j] = v_star[i][j] - dt/rho * (p_prime[i][j+1]-p_prime[i][j])/(yc[j+1]-yc[j]);

    for (int i = 1; i <= Nx; i++)
        for (int j = 1; j <= Ny; j++)
            p[i][j] = p[i][j] + p_prime[i][j] - rho*nu/2 * ((u_star[i][j]-u_star[i-1][j])/dx[i]
                                                            + (v_star[i][j]-v_star[i][j-1])/dy[j]);

    for (int j = 0; j <= Ny+1; j++) u[0][j] = u[Nx][j] = 0;
    for (int i = 0; i <= Nx; i++) {
        u[i][0] = -u[i][1];
        u[i][Ny+1] = 2-u[i][Ny];
    }
    for (int i = 0; i <= Nx+1; i++) v[i][0] = v[i][Ny] = 0;
    for (int j = 0; j <= Ny; j++) {
        v[0][j] = -v[1][j];
        v[Nx+1][j] = -v[Nx][j];
    }

    for (int i = 0; i <= Nx+1; i++) {
        p[i][0] = p[i][1];
        p[i][Ny+1] = p[i][Ny];
    }
    for (int j = 0; j <= Ny+1; j++) {
        p[0][j] = p[1][j];
        p[Nx+1][j] = p[Nx][j];
    }

    memcpy(N1_prev, N1, sizeof(N1));
    memcpy(N2_prev, N2, sizeof(N2));
}

void export(void) {
    FILE *fp;

    fp = fopen("p.txt", "w");
    if (fp) {
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny+1; j++)
                fprintf(fp, "%17.14lf ", p[i][j]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    fp = fopen("u.txt", "w");
    if (fp) {
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j <= Ny+1; j++)
                fprintf(fp, "%17.14lf ", u[i][j]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    fp = fopen("v.txt", "w");
    if (fp) {
        for (int i = 0; i <= Nx+1; i++) {
            for (int j = 0; j <= Ny; j++)
                fprintf(fp, "%17.14lf ", v[i][j]);
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}
