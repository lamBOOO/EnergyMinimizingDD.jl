#!/usr/bin/env julia
# Unit test for nonoverlapping domain decomposition

using Test

# Mock types to avoid needing full Gridap installation
# This is a simplified test that validates the logic

# Test the nonoverlapping logic in isolation
@testset "Nonoverlapping logic" begin
    println("Testing nonoverlapping domain decomposition logic...")
    
    # Simulate what create_dofs_partition does
    # freenodesp represents nodes in each subdomain before removing duplicates
    
    # Test case 1: Two subdomains with overlapping nodes
    freenodesp_original = [
        Int32[1, 2, 3, 4],  # Subdomain 1
        Int32[3, 4, 5, 6]   # Subdomain 2
    ]
    
    # Make a copy to test nonoverlapping mode
    freenodesp = deepcopy(freenodesp_original)
    
    # Apply nonoverlapping logic
    nonoverlapping = true
    if nonoverlapping
        assigned_nodes = Set{Int32}()
        for ipar = 1:length(freenodesp)
            freenodesp[ipar] = filter(node -> !(node in assigned_nodes), freenodesp[ipar])
            union!(assigned_nodes, freenodesp[ipar])
        end
    end
    
    # After nonoverlapping, nodes 3 and 4 should only be in subdomain 1
    @test length(intersect(freenodesp[1], freenodesp[2])) == 0
    @test freenodesp[1] == Int32[1, 2, 3, 4]
    @test freenodesp[2] == Int32[5, 6]
    
    # Verify all nodes are still covered
    all_nodes_original = sort(unique(vcat(freenodesp_original...)))
    all_nodes_after = sort(unique(vcat(freenodesp...)))
    @test all_nodes_original == all_nodes_after
    
    println("✓ Two subdomains with overlap test passed")
    
    # Test case 2: Three subdomains
    freenodesp_original = [
        Int32[1, 2, 3],      # Subdomain 1
        Int32[3, 4, 5],      # Subdomain 2
        Int32[5, 6, 7]       # Subdomain 3
    ]
    
    freenodesp = deepcopy(freenodesp_original)
    
    # Apply nonoverlapping logic
    if nonoverlapping
        assigned_nodes = Set{Int32}()
        for ipar = 1:length(freenodesp)
            freenodesp[ipar] = filter(node -> !(node in assigned_nodes), freenodesp[ipar])
            union!(assigned_nodes, freenodesp[ipar])
        end
    end
    
    # After nonoverlapping, no subdomain should share nodes
    @test length(intersect(freenodesp[1], freenodesp[2])) == 0
    @test length(intersect(freenodesp[2], freenodesp[3])) == 0
    @test length(intersect(freenodesp[1], freenodesp[3])) == 0
    
    @test freenodesp[1] == Int32[1, 2, 3]
    @test freenodesp[2] == Int32[4, 5]
    @test freenodesp[3] == Int32[6, 7]
    
    # Verify all nodes are still covered
    all_nodes_original = sort(unique(vcat(freenodesp_original...)))
    all_nodes_after = sort(unique(vcat(freenodesp...)))
    @test all_nodes_original == all_nodes_after
    
    println("✓ Three subdomains with overlap test passed")
    
    # Test case 3: No overlap initially (should not change anything)
    freenodesp_original = [
        Int32[1, 2],        # Subdomain 1
        Int32[3, 4]         # Subdomain 2
    ]
    
    freenodesp = deepcopy(freenodesp_original)
    
    # Apply nonoverlapping logic
    if nonoverlapping
        assigned_nodes = Set{Int32}()
        for ipar = 1:length(freenodesp)
            freenodesp[ipar] = filter(node -> !(node in assigned_nodes), freenodesp[ipar])
            union!(assigned_nodes, freenodesp[ipar])
        end
    end
    
    # Should remain the same
    @test freenodesp[1] == Int32[1, 2]
    @test freenodesp[2] == Int32[3, 4]
    @test length(intersect(freenodesp[1], freenodesp[2])) == 0
    
    println("✓ No overlap initially test passed")
    
    println("\nAll nonoverlapping logic tests passed! ✓")
end
