@testset "ASTSanitizer" begin
    @testset "safe expressions pass" begin
        san = ASTSanitizer()
        @test sanitize(san, :(y = x + Float32(1.0)))
        @test sanitize(san, :(y = sin(x)))
        @test sanitize(san, :(if x > 0.0f0; y = x; else; y = -x; end))
        @test sanitize(san, :(for i = 1:10; y = y + x; end))
    end

    @testset "unsafe calls rejected" begin
        san = ASTSanitizer()
        @test !sanitize(san, :(run(`ls`)))
        @test !sanitize(san, :(open("file.txt")))
        @test !sanitize(san, :(eval(:(1+1))))
        @test !sanitize(san, :(read("file.txt", String)))
    end

    @testset "qualified calls rejected" begin
        san = ASTSanitizer()
        @test !sanitize(san, :(Base.run(`ls`)))
        @test !sanitize(san, :(Sys.exit(1)))
    end

    @testset "macro calls rejected" begin
        san = ASTSanitizer()
        @test !sanitize(san, Expr(:macrocall, Symbol("@eval"), nothing, :(1+1)))
    end

    @testset "custom whitelist" begin
        san = ASTSanitizer(allowed_calls=Set([:+, :-, :myop]))
        @test sanitize(san, :(y = x + 1))
        @test sanitize(san, :(y = myop(x)))
        @test !sanitize(san, :(y = sin(x)))
    end

    @testset "body vector sanitization" begin
        san = ASTSanitizer()
        safe_body = [:(y = x + 1.0f0), :(y = sin(y))]
        @test sanitize(san, safe_body)
        unsafe_body = [:(y = x + 1.0f0), :(open("hack"))]
        @test !sanitize(san, unsafe_body)
    end

    @testset "Arborist boolean operators in whitelist" begin
        san = ASTSanitizer()
        @test sanitize(san, :(y = gp_nand(x, true)))
        @test sanitize(san, :(y = gp_nor(x, false)))
    end
end
