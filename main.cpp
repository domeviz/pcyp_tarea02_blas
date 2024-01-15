#include <vector>
#include <cmath>
#include <fmt/core.h>
#include "fmt/format.h"

// ASUM
// Función para calcular la suma de magnitudes de un vector
float sasum(const std::vector<float>& x) {
    float sum = 0.0f;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < x.size(); ++i) {
        sum += std::abs(x[i]);
    }

    return sum;
}

// AXPY
// Función para realizar la operación y = a * x + y
void saxpy(std::vector<float>& y, const float a, const std::vector<float>& x) {
#pragma omp parallel for
    for (int i = 0; i < y.size(); ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// COPY
// Función para copiar elementos de un vector a otro
void scopy(const std::vector<float>& x, std::vector<float>& y) {
#pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        y[i] = x[i];
    }
}

// DOT
// Función para calcular el producto punto entre dos vectores
float sdot(const std::vector<float>& x, const std::vector<float>& y) {
    float result = 0.0f;

#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }

    return result;
}

// SDSDOT
// Función para realizar el producto punto con acumulador de doble precisión
float sdsdot(const std::vector<float>& x, const std::vector<float>& y, const float s) {
    double result = static_cast<double>(s);

#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < x.size(); ++i) {
        result += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }

    return static_cast<float>(result);
}

// DOTC
// Función para realizar el producto punto conjugado
float sdotc(const std::vector<float>& x, const std::vector<float>& y) {
    float result = 0.0f;

#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }

    return result;
}

// DOTU
// Función para realizar el producto punto no conjugado
float sdotu(const std::vector<float>& x, const std::vector<float>& y) {
    float result = 0.0f;

#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < x.size(); ++i) {
        result += x[i] * y[i];
    }

    return result;
}

// NRM2
// Función para calcular la norma Euclidiana
float snrm2(const std::vector<float>& x) {
    float result = 0.0f;

#pragma omp parallel for reduction(+:result)
    for (int i = 0; i < x.size(); ++i) {
        result += x[i] * x[i];
    }

    return std::sqrt(result);
}

// SCAL
// Función para obtener el producto vector-escalar
void sscal(std::vector<float>& x, const float a) {
#pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        x[i] = a * x[i];
    }
}

// SWAP
// Función para intercambiar elementos de dos vectores
void sswap(std::vector<float>& x, std::vector<float>& y) {
#pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        std::swap(x[i], y[i]);
    }
}

// IAMAX
// Función para encontrar el índice del máximo absoluto en un vector
int isamax(const std::vector<float>& x) {
    float maxVal = std::abs(x[0]);
    int maxIndex = 0;

#pragma omp parallel for
    for (int i = 1; i < x.size(); ++i) {
        float absVal = std::abs(x[i]);
#pragma omp critical
        {
            if (absVal > maxVal) {
                maxVal = absVal;
                maxIndex = i;
            }
        }
    }

    return maxIndex;
}

// IAMIN
// Función para encontrar el índice del mínimo absoluto en un vector
int isamin(const std::vector<float>& x) {
    float minVal = std::abs(x[0]);
    int minIndex = 0;

#pragma omp parallel for
    for (int i = 1; i < x.size(); ++i) {
        float absVal = std::abs(x[i]);
#pragma omp critical
        {
            if (absVal < minVal) {
                minVal = absVal;
                minIndex = i;
            }
        }
    }

    return minIndex;
}

// ROT
// Función para realizar una rotación en el espacio bidimensional
void srot(std::vector<float>& x, std::vector<float>& y, const float c, const float s) {
#pragma omp parallel for
    for (int i = 0; i < x.size(); ++i) {
        const float temp = c * x[i] + s * y[i];
        y[i] = -s * x[i] + c * y[i];
        x[i] = temp;
    }
}

int main() {
    const int size = 5;
    const float a = 2.0f;

    std::vector<float> x(size, 1.0f);
    std::vector<float> y(size, 2.0f);
    std::vector<float> z={6,1,2,4,9,3};

    fmt::println("VECTORES");

    fmt::println("Vector x: {}", fmt::join(x, " "));
    fmt::println("Vector y: {}", fmt::join(y, " "));
    fmt::println("Vector z: {}", fmt::join(z, " "));

    float asumResultX = sasum(x);
    float asumResultY = sasum(y);
    float asumResultZ = sasum(z);

    float dotResult = sdot(x, y);
    float sdsdotResult = sdsdot(x, y, 0.5f);
    float dotcResult = sdotc(x, y);
    float dotuResult = sdotu(x, y);
    float nrm2ResultX = snrm2(x);
    float nrm2ResultY = snrm2(y);
    float nrm2ResultZ = snrm2(z);

    saxpy(y, a, x);
    scopy(x, y);
    sscal(x, a);
    sswap(x, y);

    int maxIndex = isamax(z);
    int minIndex = isamin(z);

    const float c = 0.5f;
    const float s = 0.8660254f;

    srot(x, y, c, s);

    fmt::println(" ");
    fmt::println("OPERACIONES");
    fmt::println("ASUM del Vector x: {}", asumResultX);
    fmt::println("ASUM del Vector y: {}", asumResultY);
    fmt::println("ASUM del Vector z: {}", asumResultZ);
    fmt::println("AXPY de y = a * x + y: {}", fmt::join(y, " "));
    fmt::println("DOT: {}", dotResult);
    fmt::println("SDSDOT: {}", sdsdotResult);
    fmt::println("DOTC: {}", dotcResult);
    fmt::println("DOTU: {}", dotuResult);
    fmt::println("NRM2 del Vector x: {}", nrm2ResultX);
    fmt::println("NRM2 del Vector y: {}", nrm2ResultY);
    fmt::println("NRM2 del Vector z: {}", nrm2ResultZ);
    fmt::println("SCAL: {}", fmt::join(x, " "));
    fmt::println("SWAP: {} {}", fmt::join(x, " "), fmt::join(y, " "));
    fmt::println("IAMAX del Vector z: {}", maxIndex);
    fmt::println("IAMIN del Vector z: {}", minIndex);
    fmt::println("ROT del Vector x: {}", fmt::join(x, " "));
    fmt::println("ROY del Vector y: {}", fmt::join(y, " "));
    fmt::println("ROY del Vector z: {}", fmt::join(z, " "));

    return 0;
}
